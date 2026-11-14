# -*- coding: utf-8 -*-
"""Barres en acier (module « Choix de profilé ») — la SAISIE : les quatre rubriques de la barre ouverte, dans la colonne
de gauche de l'établi compact (api_web/profil.py les pose sous la ligne de la barre).

    1 Barre                       le mode (barre chargée, efforts imposés), le modèle, les portées, le poids propre,
                                  l'acier, le profilé (il ouvre le sélecteur)
    2 Charges | Efforts           les cas de charge (nature, effort normal, charges) et leur niveau d'application ;
                                  ou les efforts de service et de calcul, et la forme des diagrammes
    3 Maintiens et flambement     les sections tenues, les longueurs de flambement (vides : automatiques)
    4 Méthodes (secondaire)       historique, Eurocode 3 ou les deux ; s, C, déversement ; γ_M0, γ_M1

Ce fichier MET EN FORME : aucun calcul. Les chemins (``data-chemin``) sont ceux que ``profil.modifier`` lit.
"""
from typing import Dict, List, Optional

from acier.barre import entree as EN
from acier.barre import parametres as PA
from acier.flambement import normes as NO
from rdm import poutres as PO

from . import commun as C
from . import etabli_compact as E
from . import profil_mots as M

ROLE = "profil"
N = C.NBSP
AIDE_JEU = {"service": "Efforts non pondérés : ceux de la méthode historique.", "elu": "Efforts de calcul (pondérés) : ceux de l'Eurocode 3."}


def _retirer(chemin: List, titre: str) -> str:
    """Une icône « retirer » en bout de ligne (le chemin finit par « - », comme etabli_compact.retirer)."""
    return (f'<button type="button" class="et-ret pf-ret" data-chemin="{C._attr_chemin(chemin)}" data-valeur="1" title="{C.esc(titre)}" '
            f'aria-label="{C.esc(titre)}">{C.icon("trash-2")}</button>')


def _titre(texte: str) -> str:
    return f'<div class="grp">{C.esc(texte)}</div>'


def _longueur(el: Dict) -> float:
    return float(sum(el["travees"]))


# ------------------------------------------------------------------ 1 · la barre
def _rubrique_barre(el: Dict, r: Dict) -> str:
    s = C.segments(["mode"], el["mode"], M.MODES)
    if el["mode"] == "chargee":
        s += C.ligne("Modèle", C.choix(["modele"], el["modele"], M.MODELES), full=True)
        if el["modele"] == "continue":
            for i, L in enumerate(el["travees"]):
                retirer = _retirer(["travees", i, "-"], "Retirer cette travée") if len(el["travees"]) > 2 else ""
                s += (f'<div class="row pf-trav"><label><span class="sym">L<sub>{i + 1}</sub></span><span>travée {i + 1}</span></label>'
                      f'{C.num(["travees", i], L, "m", dec=3, mini=EN.L_MIN, maxi=EN.L_MAX)}{retirer}</div>')
            if len(el["travees"]) < PO.TRAVEES_MAX:
                s += f'<div class="et-pied">{E.plus(["travees", "+"], "1", "Travée")}</div>'
        else:
            s += C.ligne("portée" if el["modele"] != "console" else "longueur", C.num(["travees", 0], el["travees"][0], "m", dec=3, mini=EN.L_MIN, maxi=EN.L_MAX), "L")
        pp = f" ({C.fr(r['masse'] * 9.81 / 1000.0, 2)}{N}kN/m)" if r.get("masse") else ""
        s += C.case(["poids_propre"], el["poids_propre"], f"Poids propre du profilé{pp}")
    s += _titre("Profilé")
    s += C.ligne("Acier", C.segments(["nuance"], el["nuance"], [(k, k) for k in NO.NUANCES]), full=True)
    masse = f"{M.n(r['masse'], 1)}{N}kg/m" if r.get("masse") else ""
    s += (f'<button type="button" class="pf-prof" data-pk="{ROLE}" title="Choisir le profilé dans la liste (★ : le plus léger qui tient)">'
          f'<b>{C.esc(el["profil"])}</b><span>{masse}</span>{C.icon("chevron-down")}</button>')
    return s


def _resume_barre(el: Dict) -> str:
    if el["mode"] == "efforts":
        return f"{C.esc(el['profil'])} · {C.esc(el['nuance'])} · efforts imposés"
    L = " + ".join(M.m(x) for x in el["travees"])
    return f"{C.esc(el['profil'])} · {C.esc(el['nuance'])} · {C.esc(M.NOM_MODELE[el['modele']].lower())} · {L}{N}m"


# ------------------------------------------------------------------ 2 · les charges (barre chargée)
def _charge(i: int, j: int, ch: Dict) -> str:
    """Une charge d'un cas, sur deux lignes : son type et sa valeur ; puis où elle est."""
    base = ["cas", i, "charges", j]
    t = ch["type"]
    retirer = _retirer(base + ["-"], "Retirer cette charge")
    l1 = (f'<div class="pf-ch-1">{C.choix(base + ["type"], t, M.TYPES_CHARGE)}'
          f'{C.num(base + ["v"], ch["v"], M.UNITE_CHARGE[t], dec=3)}{retirer}</div>')
    if t in ("q", "tri"):
        sens = C.choix(base + ["sens"], ch.get("sens", "croissante"), M.SENS) if t == "tri" else ""
        l2 = (f'<div class="pf-ch-2{" tri" if t == "tri" else ""}"><span>de</span>{C.num(base + ["a"], ch["a"], "m", dec=3, mini=0)}'
              f'<span>à</span>{C.num(base + ["b"], ch["b"], "m", dec=3, mini=0)}{sens}</div>')
    else:
        l2 = f'<div class="pf-ch-2 un"><span>à</span>{C.num(base + ["a"], ch["a"], "m", dec=3, mini=0)}<span>de l\'appui gauche</span></div>'
    return f'<div class="pf-ch" data-charge="{j}">{l1}{l2}</div>'


def _cas(el: Dict) -> str:
    """Un cas de charge par carte : son nom, sa nature, ses charges, son effort normal."""
    out = ""
    cas = el["cas"]
    for i, c in enumerate(cas):
        base = ["cas", i]
        retirer = _retirer(base + ["-"], "Retirer ce cas de charge") if len(cas) > 1 else ""
        charges = "".join(_charge(i, j, ch) for j, ch in enumerate(c["charges"]))
        ajouts = ""
        if len(c["charges"]) < PO.CHARGES_MAX:
            ajouts = E.plus(base + ["charges", "+"], "q", "Répartie") + E.plus(base + ["charges", "+"], "P", "Ponctuelle")
        normal = C.ligne("effort normal", C.num(base + ["N"], c["N"], "kN", dec=3), "N",
                         "Effort normal de ce cas, constant le long de la barre ; positif en compression.")
        out += (f'<div class="pf-cas" data-cas="{i}"><div class="pf-cas-h">{C.texte(base + ["nom"], c["nom"], "nom")}'
                f'{C.choix(base + ["nature"], c["nature"], M.NATURES)}{retirer}</div>{charges}'
                f'<div class="et-pied">{ajouts}</div>{normal}</div>')
    if len(cas) < EN.CAS_MAX:
        out += f'<div class="et-pied">{E.plus(["cas", "+"], "1", "Cas de charge")}</div>'
    return out


def _rubrique_charges(el: Dict) -> str:
    return (_cas(el) + _titre("Niveau d'application")
            + C.segments(["niveau"], el["niveau"], M.NIVEAUX)
            + C.aide("Compte pour le déversement : une charge posée sur la semelle haute l'aggrave."))


def _resume_charges(el: Dict) -> str:
    n_ch = sum(len(c["charges"]) for c in el["cas"])
    return f"{len(el['cas'])} cas · {n_ch} charge{'s' if n_ch > 1 else ''}"


# ------------------------------------------------------------------ 2 · les efforts imposés
def _rubrique_efforts(el: Dict) -> str:
    """Les efforts de service (méthode historique) et de calcul (Eurocode 3), côte à côte ; une seule colonne quand une
    seule méthode est choisie."""
    jeux = [(j, t) for j, t, k in (("service", "Service", "historique"), ("elu", "ELU", "ec3")) if k in el["methodes"]]
    tete = '<span></span>' + "".join(f'<span class="h" title="{C.esc(AIDE_JEU[j])}">{t}</span>' for j, t in jeux)
    lignes = ""
    for k, sym, u, aide in M.EFFORTS:
        champs = "".join(C.num(["efforts", j, k], el["efforts"][j][k], u, dec=3) for j, _ in jeux)
        lignes += f'<label title="{C.esc(aide)}"><span class="sym">{M.sub(sym)}</span></label>{champs}'
    s = f'<div class="pf-eff n{len(jeux)}">{tete}{lignes}</div>'
    s += C.aide("N positif en compression ; moments et efforts tranchants en valeur absolue.")
    s += _titre("Forme des diagrammes de moment")
    for a, nom in (("y", "M<sub>y</sub>"), ("z", "M<sub>z</sub>")):
        d = el["diagramme"][a]
        s += C.ligne(nom, C.choix(["diagramme", a, "forme"], d["forme"], M.DIAGRAMMES), full=True)
        if d["forme"] == "lineaire":
            s += C.ligne("rapport des moments d'extrémité", C.num(["diagramme", a, "psi"], d["psi"], "", dec=3, mini=-1, maxi=1), "ψ",
                         "ψ = M₂/M₁, de −1 (moments opposés) à 1 (moment constant).")
    return s


def _resume_efforts(el: Dict) -> str:
    jeu = "elu" if "ec3" in el["methodes"] else "service"
    e = el["efforts"][jeu]
    return f"{'ELU' if jeu == 'elu' else 'service'} · N {M.n(e['N'], 1)} · M<sub>y</sub> {M.n(e['My'], 1)} · V<sub>z</sub> {M.n(e['Vz'], 1)}"


# ------------------------------------------------------------------ 3 · maintiens et flambement
def _auto(r: Dict, axe: str) -> str:
    """Le texte d'un champ vide : la longueur que le moteur prend (« auto · 6 »)."""
    L = (r.get("longueurs") or {}).get(axe, {}).get("L")
    return f"auto · {M.n(L, 2)}" if L is not None else "auto"


def _flambement(el: Dict, r: Dict, auto: bool) -> str:
    s = ""
    for a, nom, aide in (("y", "axe fort", "Longueur de flambement autour de l'axe fort y-y."),
                         ("z", "axe faible", "Longueur de flambement autour de l'axe faible z-z.")):
        f = el["flambement"][a]
        champ = C.num(["flambement", a, "L"], f["L"], "m", dec=3, mini=0, auto=_auto(r, a) if auto else "")
        s += C.ligne2((f"L<sub>{a}</sub>", champ, aide + (" Vide : automatique." if auto else " 0 : axe tenu, pas de flambement.")),
                      (f"u<sub>{a}</sub>", C.num(["flambement", a, "u"], f["u"], "", dec=3, mini=0),
                       "Coefficient de longueur de flambement : 1 articulé, 0,7 encastré – articulé, 0,5 bi-encastré, 2 console."))
    return s


def _rubrique_maintiens(el: Dict, r: Dict) -> str:
    s = ""
    if el["mode"] == "chargee":
        for j, mt in enumerate(el["maintiens"]):
            s += (f'<div class="pf-mt" data-maintien="{j}"><span>à</span>{C.num(["maintiens", j, "x"], mt["x"], "m", dec=3, mini=0)}'
                  f'{C.choix(["maintiens", j, "genre"], mt["genre"], [(k, t) for k, t, _ in M.GENRES])}'
                  f'{_retirer(["maintiens", j, "-"], "Retirer ce maintien")}</div>')
        if not el["maintiens"]:
            s += C.aide("Aucun maintien : la barre n'est tenue qu'à ses appuis.")
        if len(el["maintiens"]) < EN.MAINTIENS_MAX:
            s += f'<div class="et-pied">{E.plus(["maintiens", "+"], "1", "Maintien")}</div>'
        s += C.aide("Complet : déplacement latéral et torsion empêchés. Semelle : une seule semelle tenue.")
        s += _titre("Longueurs de flambement") + _flambement(el, r, True)
        s += C.aide("Vides : la portée (axe fort) et la plus grande distance entre sections tenues (axe faible).")
    else:
        s += _titre("Longueurs de flambement") + _flambement(el, r, False)
        s += _titre("Déversement")
        s += C.ligne("entre sections tenues", C.num(["Lc"], el["Lc"], "m", dec=3, mini=0, auto=_auto(r, "z")), "L<sub>c</sub>",
                     "Distance entre deux sections tenues latéralement et en torsion. Vide : la longueur de l'axe faible. 0 : pas de déversement.")
        s += C.ligne("Charges", C.segments(["niveau"], el["niveau"], M.NIVEAUX), full=True)
    return s


def _resume_maintiens(el: Dict, r: Dict) -> str:
    Ls = r.get("longueurs") or {}
    fl = " · ".join(f"L<sub>{a}</sub> {M.n(Ls[a]['L'] * Ls[a]['u'], 2)}" for a in ("y", "z") if a in Ls)
    if el["mode"] == "chargee":
        k = len(el["maintiens"])
        return f"{k} maintien{'s' if k > 1 else ''} · {fl}{N}m" if k else f"aucun maintien · {fl}{N}m"
    return f"{fl}{N}m"


# ------------------------------------------------------------------ 4 · les méthodes
def choix_methodes(el: Dict) -> str:
    return "deux" if len(el["methodes"]) == 2 else el["methodes"][0]


def _rubrique_methodes(el: Dict, r: Dict) -> str:
    s = C.segments(["methodes"], choix_methodes(el), M.CHOIX_METHODES)
    h = el["historique"]
    if "historique" in el["methodes"]:
        s += _titre("Méthode historique · NBN B 51-001")
        s += C.ligne("coefficient de sécurité", C.num(["historique", "s"], h["s"], "", dec=3, mini=1, maxi=3), "s",
                     "σ_adm = f_y / s. 1,5 pour les charges de service ; 1,33 avec le vent.")
        s += C.ligne("Coefficient C", C.segments(["historique", "C"], h["C"], M.C_MODES), full=True)
        s += C.case(["historique", "deversement"], h["deversement"], "Déversement (semelle comprimée)")
        s += C.case(["historique", "tolerance"], h["tolerance"], "Tolérance de 5 % sur σ<sub>adm</sub>")
    if "ec3" in el["methodes"]:
        s += _titre("Eurocode 3 · NBN EN 1993-1-1")
        P = el["parametres"]
        s += C.ligne2(("γ<sub>M0</sub>", C.num(["parametres", "gamma_M0"], P.get("gamma_M0"), "", dec=3, auto=f"{M.n(PA.valeur('gamma_M0'), 2)}"),
                       "Résistance des sections. Vide : la valeur de l'annexe nationale."),
                      ("γ<sub>M1</sub>", C.num(["parametres", "gamma_M1"], P.get("gamma_M1"), "", dec=3, auto=f"{M.n(PA.valeur('gamma_M1'), 2)}"),
                       "Résistance des barres aux instabilités. Vide : la valeur de l'annexe nationale."))
        k = len((r.get("methodes") or {}).get("ec3", {}).get("a_confirmer") or [])
        if k:
            s += (f'<button type="button" class="pf-lien" data-calcul="ac" title="Les paramètres de l\'annexe nationale dont la source reste à '
                  f'confirmer">{C.icon("file-text")}{k} paramètre{"s" if k > 1 else ""} à confirmer</button>')
    return s


def _resume_methodes(el: Dict) -> str:
    noms = " + ".join(M.METHODES_COURT[k].lower() for k in el["methodes"])
    return noms + (f" · s {M.n(el['historique']['s'], 2)}" if "historique" in el["methodes"] else "")


# ------------------------------------------------------------------ la colonne
def formulaire(el: Dict, r: Optional[Dict]) -> str:
    """Les quatre rubriques de la barre ouverte ; ``r`` : son résultat (les longueurs automatiques, la masse, ce qui
    reste à confirmer). Le nom se modifie sur la ligne de la barre, en tête de la colonne."""
    r = r or {}
    charge = el["mode"] == "chargee"
    return (C.section(1, "Barre", _resume_barre(el), _rubrique_barre(el, r), True, "ba")
            + (C.section(2, "Charges", _resume_charges(el), _rubrique_charges(el), False, "ch") if charge else
               C.section(2, "Efforts", _resume_efforts(el), _rubrique_efforts(el), False, "ef"))
            + C.section(3, "Maintiens et flambement" if charge else "Flambement et déversement", _resume_maintiens(el, r),
                        _rubrique_maintiens(el, r), False, "mt")
            + C.section(4, "Méthodes", _resume_methodes(el), _rubrique_methodes(el, r), False, "me", secondaire=True))

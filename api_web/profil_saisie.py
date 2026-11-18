# -*- coding: utf-8 -*-
"""Barres en acier (module « Choix de profilé ») — la SAISIE : les rubriques de la barre ouverte, dans la colonne de
gauche de l'établi compact (api_web/profil.py les pose sous la ligne de la barre).

    1 Barre                       le mode (barre chargée, efforts imposés), le modèle, les portées, les débords, les points
                                  vérifiés, le poids propre et la position de la barre, l'acier, le profilé (il ouvre le sélecteur)
    2 Charges | Efforts           les cas de charge (nature, ψ₀ saisi, courte durée, effort normal constant ou par tronçon,
                                  charges — chacune à son niveau ; une ponctuelle dit sa longueur d'appui et son entrée dans
                                  l'âme) ; ou les efforts de service et de calcul, et la forme des diagrammes
    3 Maintiens et flambement     les sections tenues (négligées dans M_cr ou non), les longueurs de flambement (vides :
                                  automatiques)
    4 Appuis (barre chargée qui a un appui simple)   posée ou tenue par l'âme ; longueur d'appui, débord en rive, contact
                                  d'extrémité (type (c)), torsion — communs ou appui par appui (EN 1993-1-5)
    5 Méthodes (secondaire)       ce que la note imprime ; méthode ELS (s, C, déversement) ; Eurocode 3 (ouvrage,
                                  combinaisons, γ_M0, γ_M1)

Un champ neuf du lot C (C3) n'apparaît que là où il change quelque chose (le moteur le dit : le type (c) d'un lieu de l'âme,
une classe de déversement « à fournir », un profil fermé, une courbe que le tableau ne donne pas). Ce fichier MET EN FORME :
aucun calcul. Les chemins (``data-chemin``) sont ceux que ``profil.modifier`` lit.
"""
from typing import Dict, List, Optional

from acier.barre import combinaisons as CO
from acier.barre import entree as EN
from acier.barre import historique_deversement as HD
from acier.barre import nuances as NU
from acier.barre import parametres as PA
from acier.barre import section as SE
from acier.barre import statique as ST
from rdm import poutres as PO

from . import commun as C
from . import etabli_compact as E
from . import profil_mots as M
from . import profil_prs as PP

ROLE = "profil"
N = C.NBSP
AIDE_JEU = {"service": f"Efforts non pondérés : ceux de la {M.minuscule(M.METHODES['historique'])}.", "elu": "Efforts de calcul (pondérés) : ceux de l'Eurocode 3."}


def _retirer(chemin: List, titre: str) -> str:
    """Une icône « retirer » en bout de ligne (le chemin finit par « - », comme etabli_compact.retirer)."""
    return (f'<button type="button" class="et-ret pf-ret" data-chemin="{C._attr_chemin(chemin)}" data-valeur="1" title="{C.esc(titre)}" '
            f'aria-label="{C.esc(titre)}">{C.icon("trash-2")}</button>')


def _titre(texte: str) -> str:
    return f'<div class="grp">{C.esc(texte)}</div>'


def _longueur(el: Dict) -> float:
    return float(ST.longueur(el))


def forme(el: Dict) -> Optional[str]:
    """La forme de la section (acier/barre/section.FAMILLES : « I », « U », « L », un tube) ; un PRS est un I."""
    if el.get("prs"):
        return "I"
    fam = SE.CAT.par_nom().get(el["profil"], {}).get("famille")
    return SE.FAMILLES.get(fam, (None,))[0]


def _methode(r: Dict, k: str) -> Dict:
    return (r.get("methodes") or {}).get(k) or {}


def _alerte(r: Dict, code: str) -> bool:
    """La méthode ELS a-t-elle rendu l'alerte ``code`` (« a_fournir » : une classe du tableau 12 à fournir) ?"""
    return any(a.get("code") == code for a in _methode(r, "historique").get("alertes") or [])


def _cas_deversement(r: Dict) -> Optional[str]:
    """Le cas du déversement de la méthode ELS que le moteur nomme (« classeurs », « I », « ferme », « a_fournir »…)."""
    return ((_methode(r, "historique").get("grandeurs") or {}).get("deversement") or {}).get("cas")


def sans_courbe(r: Dict) -> bool:
    """Le tableau des courbes de la méthode ELS (doc p. 4) ne donne-t-il pas de courbe à ce profil ? Le moteur le dit : son
    refus (« la norme ne donne pas de courbe »), une courbe de la norme absente, ou l'alerte « courbe » d'un creux formé à
    froid de plus de 6 mm (la courbe c gardée, hors domaine)."""
    h = _methode(r, "historique")
    if "courbe" in str(h.get("refus") or ""):
        return True
    g = h.get("grandeurs") or {}
    if any(isinstance(g.get(a), dict) and g[a].get("courbe_norme") is None for a in ("y", "z")):
        return True
    return any(a.get("code") == "courbe" and "froid" in (a.get("message") or "") for a in h.get("alertes") or [])


def lieux_c(r: Dict, genre: str) -> List[float]:
    """Les abscisses (m) des lieux de l'âme du TYPE (c) — ``genre`` « appui » ou « charge » : le moteur dit où le contact
    d'extrémité compte (brief § 8.2)."""
    return [L["x"] for L in (r.get("ame") or {}).get("lieux") or [] if L.get("genre") == genre and L.get("type") == "c"]


def _proche(x: float, xs: List[float]) -> bool:
    return any(abs(x - y) < 1e-6 for y in xs)


# ------------------------------------------------------------------ 1 · la barre
def _rubrique_barre(el: Dict, r: Dict, carnet: Optional[List[Dict]] = None) -> str:
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
        if el["modele"] in PO.MODELES_DEBORDS:              # (lot C, C7) les porte-à-faux : deux appuis et continue seulement
            db = el["debords"]
            s += C.ligne2(("débord ←", C.num(["debords", 0], db[0], "m", dec=3, mini=0, maxi=EN.L_MAX), M.AIDE_DEBORDS),
                          ("→", C.num(["debords", 1], db[1], "m", dec=3, mini=0, maxi=EN.L_MAX), ""))
        for k, x in enumerate(el["points"]):                # (lot C) les sections vérifiées en plus (brief § 10)
            s += (f'<div class="row pf-trav"><label><span>point vérifié</span></label>{C.num(["points", k], x, "m", dec=3, mini=0)}'
                  f'{_retirer(["points", k, "-"], "Retirer ce point")}</div>')
        if len(el["points"]) < EN.POINTS_MAX:
            s += f'<div class="et-pied">{E.plus(["points", "+"], "1", "Point vérifié")}</div>'
        pp = f" ({C.fr(r['poids_propre'], 2)}{N}kN/m)" if r.get("poids_propre") else ""          # la valeur que le moteur compte
        s += C.case(["poids_propre"], el["poids_propre"], f"Poids propre du profilé{pp}")
        if el["poids_propre"]:                               # la position ne compte que pour le poids propre (brief § 4.1)
            s += C.ligne("Position", C.segments(["position"], el["position"], M.POSITIONS), full=True)
            if el["position"] == "inclinee":
                s += C.ligne("angle sur l'horizontale", C.num(["angle"], el["angle"], "°", dec=1, mini=0, maxi=90), "α")
    else:                                                    # (lot C) efforts imposés : le poids propre, deux interrupteurs
        pp = f" ({C.fr(r['poids_propre'], 2)}{N}kN/m)" if r.get("poids_propre") else ""
        ppe = el["poids_propre_efforts"]
        s += (f'<div class="row pf-pp"><label><span>Poids propre{pp}</span></label>'
              f'{C.case(["poids_propre_efforts", "N"], ppe["N"], "dans P")}{C.case(["poids_propre_efforts", "M"], ppe["M"], "en flexion")}</div>')
    s += _titre("Profilé")
    s += C.ligne("Acier", C.choix(["nuance"], el["nuance"], [(k, k) for k in NU.NUANCES]), full=True)   # un menu : cinq nuances (C1b)
    if el["nuance"] in NU.AVEC_NORME:                        # (lot C, C1b) la norme produit ne change que S420 et S460
        s += C.ligne("norme produit", C.segments(["norme_produit"], el["norme_produit"] or "",
                                                 [("", M_NORME[None][0], M_NORME[None][1]), ("M", M_NORME["M"][0], M_NORME["M"][1])]), full=True)
    # (lot C, C6) « Catalogue · PRS », les plats d'un PRS et sa bibliothèque, le bouton du profilé : api_web/profil_prs.py
    return s + PP.rubrique(el, r, carnet)


# la norme produit de S420 et S460 : (mot, survol) — les noms de acier/barre/nuances.py
M_NORME = {None: (NU.LIBELLES_NORME[None], "EN 10025-3 (normalisé)"), "M": (NU.LIBELLES_NORME["M"], "EN 10025-4 (thermomécanique)")}


def _resume_barre(el: Dict) -> str:
    if el["mode"] == "efforts":
        return f"{C.esc(el['profil'])} · {C.esc(el['nuance'])} · efforts imposés"
    L = " + ".join(M.m(x) for x in el["travees"])
    return f"{C.esc(el['profil'])} · {C.esc(el['nuance'])} · {C.esc(M.NOM_MODELE[el['modele']].lower())} · {L}{N}m"


# ------------------------------------------------------------------ 2 · les charges (barre chargée)
def _niveau_barre(el: Dict) -> str:
    """Le niveau d'application des charges de la barre — écrit une fois : en barre chargée, le niveau PAR DÉFAUT (chaque
    charge peut avoir le sien) ; en efforts imposés, celui des charges."""
    nom = "niveau par défaut" if el["mode"] == "chargee" else "niveau des charges"
    return C.ligne(nom, C.segments(["niveau"], el["niveau"], M.NIVEAUX), full=True)


def _charge(i: int, j: int, ch: Dict, el: Dict, contacts: List[float]) -> str:
    """Une charge d'un cas : son type, sa valeur, son niveau (▲ ● ▼ ; à défaut celui de la barre) ; puis où elle est. Une
    ponctuelle qui appuie sur une semelle (hors du centre de cisaillement) dit aussi sa longueur d'appui, son entrée dans
    l'âme et, sur un lieu du type (c), le contact d'extrémité."""
    base = ["cas", i, "charges", j]
    t = ch["type"]
    niv = ch.get("niveau") or el["niveau"]
    retirer = _retirer(base + ["-"], "Retirer cette charge")
    niveau = f'<div class="pf-niv">{C.segments(base + ["niveau"], niv, M.NIVEAUX_CHARGE)}</div>' if t != "C" else ""
    l1 = (f'<div class="pf-ch-1{" niv" if niveau else ""}">{C.choix(base + ["type"], t, M.TYPES_CHARGE)}'
          f'{C.num(base + ["v"], ch["v"], M.UNITE_CHARGE[t], dec=3)}{niveau}{retirer}</div>')
    appuie = t == "P" and niv != "centre"
    plus = ""
    if t in ("q", "tri"):
        sens = C.choix(base + ["sens"], ch.get("sens", "croissante"), M.SENS) if t == "tri" else ""
        l2 = (f'<div class="pf-ch-2{" tri" if t == "tri" else ""}"><span>de</span>{C.num(base + ["a"], ch["a"], "m", dec=3, mini=0)}'
              f'<span>à</span>{C.num(base + ["b"], ch["b"], "m", dec=3, mini=0)}{sens}</div>')
    elif appuie:
        l2 = (f'<div class="pf-ch-2 ss"><span>à</span>{C.num(base + ["a"], ch["a"], "m", dec=3, mini=0)}'
              f'<span title="{C.esc(M.AIDE_SS)}">sur</span>{C.num(base + ["ss"], ch.get("ss"), "mm", dec=0, mini=0, maxi=EN.SS_MAX, auto=M.SS_VIDE)}</div>')
        plus = C.ligne("entrée dans l'âme", C.segments(base + ["introduction"], ch.get("introduction", "auto"), M.INTRODUCTIONS), full=True)
        if _proche(ch["a"], contacts):
            plus += C.ligne("contact", C.choix(base + ["contact"], ch.get("contact", "inconnu"), M.CONTACTS), "", M.AIDE_CONTACT)
    else:
        depuis = "du bout gauche" if el["debords"][0] else "de l'appui gauche"
        l2 = f'<div class="pf-ch-2 un"><span>à</span>{C.num(base + ["a"], ch["a"], "m", dec=3, mini=0)}<span>{depuis}</span></div>'
    return f'<div class="pf-ch" data-charge="{j}">{l1}{l2}{plus}</div>'


def _normal(i: int, c: Dict, el: Dict) -> str:
    """L'effort normal d'un cas : constant, ou par tronçon (chaque tronçon jusqu'à son abscisse, le dernier jusqu'au bout)."""
    base = ["cas", i]
    nt = c.get("N_troncons")
    seg = C.segments(base + ["N_troncons"], "troncons" if nt else "constant", M.N_MODES)
    champ = "" if nt else C.num(base + ["N"], c["N"], "kN", dec=3)
    s = (f'<div class="row pf-n"><label title="Effort normal ; positif en compression."><span class="sym">N</span></label>'
         f'{seg}{champ}</div>')
    if not nt:
        return s
    for k, t in enumerate(nt):
        b = base + ["N_troncons", k]
        fin = (C.num(b + ["x2"], t["x2"], "m", dec=3, mini=0) if k < len(nt) - 1 else '<span class="pf-bout">au bout</span>')
        retirer = _retirer(b + ["-"], "Retirer ce tronçon") if len(nt) > 1 else ""
        s += f'<div class="pf-nt"><span>jusqu\'à</span>{fin}{C.num(b + ["N"], t["N"], "kN", dec=3)}{retirer}</div>'
    return s + f'<div class="et-pied">{E.plus(base + ["N_troncons", "+"], "1", "Tronçon")}</div>'


def _cas(el: Dict, r: Dict) -> str:
    """Un cas de charge par carte : son nom, sa nature (et son ψ₀ quand il est saisi), ses charges, sa durée quand elle
    compte, son effort normal."""
    out = ""
    cas = el["cas"]
    contacts = lieux_c(r, "charge")
    # la case « courte durée » ne change rien sans neige ni vent, ni quand la règle est désactivée (brief § 5)
    courte = (any(c["nature"] in CO.COURTES for c in cas) and el["combinaisons"]["regle_courte_duree"])
    for i, c in enumerate(cas):
        base = ["cas", i]
        retirer = _retirer(base + ["-"], "Retirer ce cas de charge") if len(cas) > 1 else ""
        charges = "".join(_charge(i, j, ch, el, contacts) for j, ch in enumerate(c["charges"]))
        ajouts = ""
        if len(c["charges"]) < PO.CHARGES_MAX:
            ajouts = E.plus(base + ["charges", "+"], "q", "Répartie") + E.plus(base + ["charges", "+"], "P", "Ponctuelle")
        extra = ""
        if c["nature"] in CO.PSI0_SAISI:                     # (lot C, C2) exécution, machines : ψ₀ saisi, sans défaut
            extra += C.ligne("combinaison", C.num(base + ["psi0"], c.get("psi0"), "", dec=2, mini=0, maxi=1, auto="à saisir"),
                             "ψ<sub>0</sub>", CO.AIDE_PSI0[c["nature"]])
        if courte and c["nature"] != CO.PERMANENT and c["nature"] not in CO.COURTES:
            extra += (f'<span title="{C.esc(M.AIDE_COURTE_DUREE)}">'
                      f'{C.case(base + ["courte_duree"], c.get("courte_duree", False), "de courte durée (moins d&#39;un mois)")}</span>')
        out += (f'<div class="pf-cas" data-cas="{i}"><div class="pf-cas-h">{C.texte(base + ["nom"], c["nom"], "nom")}'
                f'{C.choix(base + ["nature"], c["nature"], M.NATURES)}{retirer}</div>{extra}{charges}'
                f'<div class="et-pied">{ajouts}</div>{_normal(i, c, el)}</div>')
    if len(cas) < EN.CAS_MAX:
        out += f'<div class="et-pied">{E.plus(["cas", "+"], "1", "Cas de charge")}</div>'
    return out


def _rubrique_charges(el: Dict, r: Dict) -> str:
    return _cas(el, r) + _niveau_barre(el)


def _resume_charges(el: Dict) -> str:
    n_ch = sum(len(c["charges"]) for c in el["cas"])
    return f"{len(el['cas'])} cas · {n_ch} charge{'s' if n_ch > 1 else ''}"


# ------------------------------------------------------------------ 2 · les efforts imposés
def _rubrique_efforts(el: Dict) -> str:
    """Les efforts de service (méthode ELS) et de calcul (Eurocode 3), côte à côte — toujours les deux colonnes : les deux
    méthodes sont toujours calculées (lot C, C0d)."""
    jeux = [("service", "Service"), ("elu", "ELU")]
    tete = '<span></span>' + "".join(f'<span class="h" title="{C.esc(AIDE_JEU[j])}">{t}</span>' for j, t in jeux)
    lignes = ""
    for k, sym, u, aide in M.EFFORTS:
        champs = "".join(C.num(["efforts", j, k], el["efforts"][j][k], u, dec=3) for j, _ in jeux)
        # (intégration du lot C, R1) l'aide dans l'icône .aide[title], lisible au doigt (tablette, téléphone) comme au survol
        lignes += f'<label><span class="sym">{M.sub(sym)}</span><span class="aide" title="{C.esc(aide)}">{C.icon("info")}</span></label>{champs}'
    s = f'<div class="pf-eff n{len(jeux)}">{tete}{lignes}</div>'
    s += _titre("Forme des diagrammes de moment")
    for a, nom in (("y", "M<sub>y</sub>"), ("z", "M<sub>z</sub>")):
        d = el["diagramme"][a]
        s += C.ligne(nom, C.choix(["diagramme", a, "forme"], d["forme"], M.DIAGRAMMES), full=True)
        if d["forme"] == "lineaire":
            s += C.ligne("rapport des moments d'extrémité", C.num(["diagramme", a, "psi"], d["psi"], "", dec=3, mini=-1, maxi=1), "ψ",
                         "ψ = M₂/M₁, de −1 (moments opposés) à 1 (moment constant).")
    return s


def _resume_efforts(el: Dict) -> str:
    e = el["efforts"]["elu"]
    return f"ELU · N {M.n(e['N'], 1)} · M<sub>y</sub> {M.n(e['My'], 1)} · V<sub>z</sub> {M.n(e['Vz'], 1)}"


# ------------------------------------------------------------------ 3 · maintiens et flambement
def _auto(r: Dict, axe: str) -> str:
    """Le texte d'un champ vide : la longueur que le moteur prend (« auto · 6 »)."""
    L = (r.get("longueurs") or {}).get(axe, {}).get("L")
    return f"auto · {M.n(L, 2)}" if L is not None else "auto"


def _flambement(el: Dict, r: Dict, auto: bool) -> str:
    s = ""
    for a, nom in (("y", "Longueur de flambement autour de l'axe fort y-y."), ("z", "Longueur de flambement autour de l'axe faible z-z.")):
        f = el["flambement"][a]
        champ = C.num(["flambement", a, "L"], f["L"], "m", dec=3, mini=0, auto=_auto(r, a) if auto else "")
        if auto:                                             # (lot C, C1) l'origine de la longueur proposée
            origine = M.ORIGINES.get(((r.get("longueurs") or {}).get(a) or {}).get("origine"))
            aide = nom + (f" Vide : {origine}." if origine else " Vide : automatique.")
        else:
            aide = nom + " 0 : axe tenu, pas de flambement."
        s += C.ligne2((f"L<sub>{a}</sub>", champ, aide), (f"u<sub>{a}</sub>", C.num(["flambement", a, "u"], f["u"], "", dec=3, mini=0), M.AIDE_U))
    return s


def _courbes(el: Dict, r: Dict) -> str:
    """(lot C) La courbe de flambement par axe, en efforts imposés, seulement quand le tableau n'en donne pas — ou quand
    une courbe est déjà imposée (elle reste lisible et se retire)."""
    fl = el["flambement"]
    if not ((el["mode"] == "efforts" and sans_courbe(r)) or any(fl[a].get("courbe") for a in ("y", "z"))):
        return ""
    return C.ligne2(*((f"courbe {a}", C.choix(["flambement", a, "courbe"], fl[a].get("courbe") or "", M.COURBES_AXE),
                       M.AIDE_COURBE if a == "y" else "") for a in ("y", "z")))


def _rubrique_maintiens(el: Dict, r: Dict) -> str:
    s = ""
    if el["mode"] == "chargee":
        for j, mt in enumerate(el["maintiens"]):
            s += (f'<div class="pf-mt" data-maintien="{j}"><span>à</span>{C.num(["maintiens", j, "x"], mt["x"], "m", dec=3, mini=0)}'
                  f'{C.choix(["maintiens", j, "genre"], mt["genre"], [(k, t) for k, t, _ in M.GENRES])}'
                  f'<span class="pf-neg" title="{C.esc(M.AIDE_NEGLIGE)}">{C.case(["maintiens", j, "neglige"], mt.get("neglige", False), "négligé")}</span>'
                  f'{_retirer(["maintiens", j, "-"], "Retirer ce maintien")}</div>')
        if len(el["maintiens"]) < EN.MAINTIENS_MAX:
            s += f'<div class="et-pied">{E.plus(["maintiens", "+"], "1", "Maintien")}</div>'
        s += C.ligne("maintiens réguliers", C.texte(["maintiens", "reguliers"], "", "n · ou n a b"), "n", M.AIDE_REGULIERS)
        s += _titre("Longueurs de flambement") + _flambement(el, r, True) + _courbes(el, r)
    else:
        s += _titre("Longueurs de flambement") + _flambement(el, r, False) + _courbes(el, r)
        s += _titre("Déversement")
        s += C.ligne("entre sections tenues", C.num(["Lc"], el["Lc"], "m", dec=3, mini=0, auto=_auto(r, "z")), "L<sub>c</sub>",
                     "Distance entre deux sections tenues latéralement et en torsion. Vide : la longueur de l'axe faible. 0 : pas de déversement.")
        s += _niveau_barre(el)
    return s


def _resume_maintiens(el: Dict, r: Dict) -> str:
    Ls = r.get("longueurs") or {}
    fl = " · ".join(f"L<sub>{a}</sub> {M.n(Ls[a]['L'] * Ls[a]['u'], 2)}" for a in ("y", "z") if a in Ls)
    if el["mode"] == "chargee":
        k = len(el["maintiens"])
        return f"{k} maintien{'s' if k > 1 else ''} · {fl}{N}m" if k else f"aucun maintien · {fl}{N}m"
    return f"{fl}{N}m"


# ------------------------------------------------------------------ 4 · les appuis (ce que l'âme en reçoit)
def a_des_appuis(el: Dict) -> bool:
    """La barre chargée a-t-elle un appui SIMPLE (une console, une barre bi-encastrée n'en ont pas) ?"""
    return el["mode"] == "chargee" and el["modele"] in ("appuis", "enc_app", "continue")


def _appui(i: int, n: int, d: Dict, ap: Dict, contact: bool) -> str:
    """Un appui, en une ligne compacte : sa lettre, s_s et c (appui de rive) quand la barre est posée, sa torsion ; puis le
    contact d'extrémité quand l'appui est un lieu du type (c). Une clé absente : la valeur commune (acier/barre/ame.py)."""
    b = ["appuis", "detail", i]
    poses = ap["mode"] == "poses"
    ss = C.num(b + ["ss"], d["ss"] if "ss" in d else ap["ss"], "mm", dec=0, mini=0, maxi=EN.SS_MAX, auto=M.SS_VIDE) if poses else ""
    rive = poses and i in (0, n - 1)
    c = C.num(b + ["c"], d["c"] if "c" in d else ap["c"], "mm", dec=0, mini=0, maxi=EN.SS_MAX, auto=M.SS_VIDE) if rive else ""
    lettre = chr(65 + i)
    ss_ = f'<span title="{C.esc(M.AIDE_SS)}">s<sub>s</sub></span>{ss}' if poses else ""
    c_ = f'<span title="{C.esc(M.AIDE_DEBORD)}">c</span>{c}' if rive else ""
    s = (f'<div class="pf-ap{" poses" if poses else ""}{" rive" if rive else ""}"><span class="sym" title="Appui {lettre}">{lettre}</span>'
         f'{ss_}{c_}{C.choix(b + ["torsion"], d.get("torsion", "fourche"), M.TORSIONS)}</div>')
    if contact:
        s += C.ligne(f"contact {chr(65 + i)}", C.choix(b + ["contact"], d.get("contact", ap.get("contact", "inconnu")), M.CONTACTS), "",
                     M.AIDE_CONTACT)
    return s


def _rubrique_appuis(el: Dict, r: Dict) -> str:
    ap = el["appuis"]
    s = C.segments(["appuis", "mode"], ap["mode"], M.MODES_APPUI)
    detail = ap.get("detail")
    s += C.ligne("appuis", C.segments(["appuis", "detail"], "un_par_un" if detail is not None else "identiques", M.DETAIL_APPUIS), full=True)
    xs = ST.appuis_x(el)
    contacts = lieux_c(r, "appui")
    if detail is None:
        if ap["mode"] == "poses":
            s += C.ligne("longueur d'appui", C.num(["appuis", "ss"], ap["ss"], "mm", dec=0, mini=0, maxi=EN.SS_MAX, auto=M.SS_VIDE), "s<sub>s</sub>", M.AIDE_SS)
            s += C.ligne("débord en rive", C.num(["appuis", "c"], ap["c"], "mm", dec=0, mini=0, maxi=EN.SS_MAX, auto=M.SS_VIDE), "c", M.AIDE_DEBORD)
        if contacts:                                       # (lot B, B4) le contact d'extrémité : seulement sur un lieu du type (c)
            s += C.ligne("contact d'extrémité", C.choix(["appuis", "contact"], ap.get("contact", "inconnu"), M.CONTACTS), "", M.AIDE_CONTACT)
        return s
    for i, x in enumerate(xs):
        d = detail[i] if i < len(detail) and detail[i] is not None else {}
        s += _appui(i, len(xs), d, ap, _proche(x, contacts))
    return s


def _resume_appuis(el: Dict) -> str:
    ap = el["appuis"]
    if ap["mode"] != "poses":
        return M.NOM_MODE_APPUI[ap["mode"]]
    if ap.get("detail") is not None:
        return f"{M.NOM_MODE_APPUI['poses']} · un par un"
    # (la rubrique n'est offerte qu'à une barre qui a un appui de rive simple : son débord c compte ; 0 saisi n'est pas vide) — un
    # champ vide se résume comme il s'affiche : « borne 0 » (M.SS_VIDE)
    ss = f"s<sub>s</sub> {M.n(ap['ss'], 0)}{N}mm" if ap["ss"] is not None else f"s<sub>s</sub> {M.SS_VIDE}"
    c = f"c {M.n(ap['c'], 0)}{N}mm" if ap["c"] is not None else f"c {M.SS_VIDE}"
    return f"{M.NOM_MODE_APPUI['poses']} · {ss} · {c}"


# ------------------------------------------------------------------ 5 · les méthodes
def _gamma(P: Dict, cle: str) -> float:
    return P.get(cle) if P.get(cle) is not None else PA.valeur(cle)


def _rubrique_methodes(el: Dict, r: Dict, impression: str) -> str:
    """Ce que la NOTE imprime (``profil.IMPRESSION`` : pour toutes ses barres, E11), puis les paramètres des deux méthodes,
    toujours calculées (lot C, C0d) — chacun seulement là où il compte."""
    s = C.ligne("Note (toutes les barres)", C.segments(["impression"], impression, M.CHOIX_METHODES), full=True)
    h = el["historique"]
    s += _titre(M.METHODES["historique"])
    s += C.ligne("coefficient de sécurité", C.num(["historique", "s"], h["s"], "", dec=3, mini=1, maxi=3), "s",
                 "σ_adm = f_y / s ; 1,5 pour les charges de service.")
    s += C.ligne("Coefficient C", C.segments(["historique", "C"], h["C"], M.C_MODES), full=True)
    if forme(el) == "I":                                  # (lot B, B1) la variante et C_d ne valent que pour un I
        s += C.ligne("Déversement", C.segments(["historique", "variante"], h["variante"], M.VARIANTES_DEV), full=True)
        if h["variante"] == "classeurs":
            s += C.ligne("coefficient de moment", C.num(["historique", "Cd"], h["Cd"], "", dec=2, mini=HD.CD_MIN, maxi=HD.CD_MAX),
                         "C<sub>d</sub>", M.AIDE_CD)
    if _alerte(r, "a_fournir") or h.get("classe_d"):      # une classe imposée : seulement quand le tableau 12 n'en donne pas
        s += C.ligne("classe de déversement", C.choix(["historique", "classe_d"], h.get("classe_d") or "",
                                                       [("", "à fournir")] + [(k, k) for k in HD.CLASSES]), "", M.AIDE_CLASSE_D)
    if _cas_deversement(r) == "ferme" or h.get("facteur_doc"):   # le facteur 5,1 de la doc : un profil fermé seulement
        s += C.case(["historique", "facteur_doc"], h.get("facteur_doc", False), "facteur 5,1 de la doc (profil fermé, variante)")
    if not h["deversement"]:                              # (une ancienne note) le déversement éteint : le rallumer
        s += C.case(["historique", "deversement"], False, "Déversement vérifié")
    s += _titre("Eurocode 3 · NBN EN 1993-1-1")
    s += C.ligne("Ouvrage", C.segments(["ouvrage"], el["ouvrage"], M.OUVRAGES), full=True)
    if el["mode"] == "chargee":
        co = el["combinaisons"]
        s += C.ligne("Combinaisons", C.segments(["combinaisons", "expression"], co["expression"], M.EXPRESSIONS), full=True)
        if co["expression"] == "6.10ab":
            s += C.case(["combinaisons", "geotechnique"], co["geotechnique"], "application géotechnique (ξ = 1)")
        if any(c["nature"] in CO.COURTES for c in el["cas"]):
            s += f'<span title="{C.esc(M.AIDE_REGLE_COURTE)}">{C.case(["combinaisons", "regle_courte_duree"], co["regle_courte_duree"], "règle de courte durée")}</span>'
    P = el["parametres"]
    s += C.ligne2(("γ<sub>M0</sub>", C.num(["parametres", "gamma_M0"], P.get("gamma_M0"), "", dec=3, auto=f"{M.n(PA.valeur('gamma_M0'), 2)}"),
                   "Résistance des sections. Vide : la valeur de l'annexe nationale."),
                  ("γ<sub>M1</sub>", C.num(["parametres", "gamma_M1"], P.get("gamma_M1"), "", dec=3, auto=f"{M.n(PA.valeur('gamma_M1'), 2)}"),
                   "Résistance des barres aux instabilités. Vide : la valeur de l'annexe nationale."))
    if abs(_gamma(P, "gamma_M0") - _gamma(P, "gamma_M1")) > 1e-12:
        # le seul paramètre « à confirmer » qui reste (parametres.npl_gamma) : dit près des champs qui le font jouer
        npl = PA.PARAMETRES["npl_gamma"]
        txt = f"{M.sub(npl['nom'])} : {M.sub(npl['valeur'].replace('gamma_', 'γ_'))}, à confirmer."
        if _methode(r, "ec3").get("a_confirmer"):
            s += f'<button type="button" class="pf-lien" data-calcul="ac" title="{C.esc(npl["source"])}">{C.icon("file-text")}{txt}</button>'
        else:
            s += C.aide(txt)
    return s


def _resume_methodes(el: Dict, impression: str = "deux") -> str:
    note = M.minuscule(dict(M.CHOIX_METHODES).get(impression, M.CHOIX_METHODES[0][1]))
    return f"note : {note} · s {M.n(el['historique']['s'], 2)}"


# ------------------------------------------------------------------ la colonne
def formulaire(el: Dict, r: Optional[Dict], impression: str = "deux", carnet: Optional[List[Dict]] = None) -> str:
    """Les rubriques de la barre ouverte ; ``r`` : son résultat (les longueurs automatiques, la masse, ce que le moteur dit
    des lieux de l'âme, du déversement et des courbes) ; ``impression`` : ce que la note imprime (``profil.IMPRESSION``) ;
    ``carnet`` : la bibliothèque des PRS (lot C, C6). Le nom se modifie sur la ligne de la barre, en tête de la colonne."""
    r = r or {}
    charge = el["mode"] == "chargee"
    appuis = a_des_appuis(el)
    return (C.section(1, "Barre", _resume_barre(el), _rubrique_barre(el, r, carnet), True, "ba")
            + (C.section(2, "Charges", _resume_charges(el), _rubrique_charges(el, r), False, "ch") if charge else
               C.section(2, "Efforts", _resume_efforts(el), _rubrique_efforts(el), False, "ef"))
            + C.section(3, "Maintiens et flambement" if charge else "Flambement et déversement", _resume_maintiens(el, r),
                        _rubrique_maintiens(el, r), False, "mt")
            + (C.section(4, "Appuis", _resume_appuis(el), _rubrique_appuis(el, r), False, "ap") if appuis else "")
            + C.section(5 if appuis else 4, "Méthodes", _resume_methodes(el, impression), _rubrique_methodes(el, r, impression), False, "me",
                        secondaire=True))

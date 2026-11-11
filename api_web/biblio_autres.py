# -*- coding: utf-8 -*-
"""La BIBLIOTHÈQUE « AUTRES » du site (docs/site/BIBLIOTHEQUE_AUTRES.md ; docs/site/BIBLIOTHEQUES.md § 2.6) : trois
OUTILS de résistance des matériaux — le formulaire de poutres, les plaques rectangulaires, les caractéristiques d'une
section quelconque. Même contrat que les autres bibliothèques (api_web/biblio_beton.py) :

    consulter(rubrique, params, hote="page") → {rubrique, bibliotheque, titre, barre, tableau, bande, pourquoi, sources,
                                                  fiche, params, message}

Ici « tableau » est le CADRE de l'outil : le dessin, les données, les résultats et la feuille de calcul (formule
générale → valeurs → résultat). Les calculs viennent des moteurs ``rdm/`` ; ce fichier porte la porte d'entrée, le
menu et les BRIQUES d'écran communes aux trois outils (feuille de calcul, bloc, bande, volet). Chaque outil a son
fichier : api_web/biblio_poutres.py, biblio_plaques.py, biblio_sections.py.
"""
import json
from typing import Dict, List, Optional, Sequence

from api_web import commun as C

BIBLIOTHEQUE = "autres"
TITRE = "Bibliothèque Autres"
PAGE = "bibliotheque-autres"
VERSION = "2026-11-11"
# (clé, titre du menu, sous-titre du menu, icône)
RUBRIQUES = (
    ("poutres", "Poutres", "Formulaire de RDM", "rdm-poutre"),
    ("plaques", "Plaques", "Plaques rectangulaires", "rdm-plaque"),
    ("sections", "Sections", "Section quelconque", "rdm-section"),
)
CLES = tuple(r[0] for r in RUBRIQUES)
NOMS = {r[0]: r[1] for r in RUBRIQUES}
RUBRIQUE_DEFAUT = "poutres"
HOTES = ("page", "note")
NBSP = C.NBSP
MOINS = "−"


# ================================================================================================ les nombres
def n(x: float, dec: int = 4) -> str:
    """Un nombre DANS une formule (« = valeurs ») : écrit tel quel jusqu'à ``dec`` décimales, le signe moins
    typographique, entre parenthèses s'il est négatif — « 10 × (−4) » se relit sans ambiguïté."""
    if abs(x) < 0.5 * 10 ** -dec:
        return "0"
    s = C.nb(abs(x), dec)
    return f"({MOINS}{s})" if x < 0 else s


def r(x: Optional[float], nd: int = 2) -> str:
    """Un RÉSULTAT : ``nd`` décimales, virgule, signe moins typographique."""
    return C.fr(x, nd).replace("-", MOINS)


def m(texte: str) -> str:
    """Un texte de formule en HTML : les indices (M_max → M<sub>max</sub>), échappé."""
    return C.symb_math(texte)


# ================================================================================================ les briques d'écran
def ligne(sym: str, nom: str = "", formule: str = "", valeurs: str = "", valeur: str = "", unite: str = "", ref: str = "",
          on: bool = False, cle: str = "") -> Dict:
    """Une ligne de la feuille de calcul : symbole, nom, formule générale, valeurs remplacées, résultat (texte), unité,
    référence. ``cle`` la retrouve (tests, impression) ; ``on`` l'allume (la valeur de la bande)."""
    return {"sym": sym, "nom": nom, "formule": formule, "valeurs": valeurs, "valeur": valeur, "unite": unite, "ref": ref, "on": on, "cle": cle}


def feuille(groupes: Sequence) -> str:
    """La feuille de calcul : des groupes (titre, sous-titre, lignes) ; chaque ligne donne la formule générale, puis
    « = valeurs », puis le résultat et son unité, puis la référence. Le tableau est celui des feuilles de la
    bibliothèque (css : .bib-fc)."""
    corps = ""
    for titre, sous, lignes in groupes:
        if not lignes:
            continue
        corps += f'<tr class="g"><th colspan="5"><b>{C.esc(titre)}</b>{(" <span>" + m(sous) + "</span>") if sous else ""}</th></tr>'
        for li in lignes:
            calc = f'<span class="fo">{m(li["formule"])}</span>' if li["formule"] else ""
            if li["valeurs"]:
                calc += f'<span class="va">= {m(li["valeurs"])}</span>'
            u = f'{NBSP}<span class="u">{m(li["unite"])}</span>' if li["unite"] else ""
            corps += (f'<tr data-terme="{C.esc(li["cle"])}"{" class=" + chr(34) + "on" + chr(34) if li["on"] else ""}>'
                      f'<th class="nm"><span class="s">{m(li["sym"])}</span></th><td class="nom">{m(li["nom"])}</td>'
                      f'<td class="fc">{calc}</td><td class="v"><b>{C.esc(li["valeur"])}</b>{u}</td>'
                      f'<td class="rg">{m(li["ref"])}</td></tr>')
    cols = '<colgroup><col class="c-s"><col class="c-n"><col class="c-c"><col class="c-v"><col class="c-r"></colgroup>'
    return f'<table class="bib-tab bib-fc rdm-fc">{cols}<tbody>{corps}</tbody></table>'


def bloc(titre: str, contenu: str, cls: str = "", outils: str = "") -> str:
    """Un bloc titré du cadre (Charges, Résultats, Feuille de calcul…)."""
    return (f'<section class="rdm-bloc{(" " + cls) if cls else ""}"><header class="rdm-bh"><h3>{titre}</h3>{outils}</header>'
            f'<div class="rdm-bc">{contenu}</div></section>')


def sel(d: Dict) -> str:
    """L'attribut data-sel d'un bouton : les paramètres qu'un clic règle (JSON échappé)."""
    return f' data-sel="{C.esc(json.dumps(d, ensure_ascii=False))}"'


def bouton(texte: str, d: Dict, icone: str = "", cls: str = "btn sm", titre: str = "") -> str:
    t = f' title="{C.esc(titre)}" aria-label="{C.esc(titre)}"' if titre else ""
    return f'<button type="button" class="{cls}"{sel(d)}{t}>{C.icon(icone) if icone else ""}{("<span>" + texte + "</span>") if texte else ""}</button>'


def champ(cle: str, valeur, unite: str = "", libelle: str = "", large: bool = False, dec: int = 4) -> str:
    """Un champ numérique DANS le cadre (js/biblio.js : tout [data-param] du cadre règle son paramètre)."""
    v = "" if valeur is None else C.nb(float(valeur), dec)
    u = f'<span class="u">{unite}</span>' if unite else ""
    lab = f'<span class="rdm-cl">{libelle}</span>' if libelle else ""
    return (f'<label class="rdm-c{" l" if large else ""}">{lab}<span class="field"><input data-param="{C.esc(cle)}" inputmode="decimal" '
            f'value="{C.esc(v)}" aria-label="{C.esc(_brut(libelle) or cle)}">{u}</span></label>')


def menu_champ(cle: str, valeur, options: Sequence, libelle: str = "") -> str:
    opts = "".join(f'<option value="{C.esc(str(k))}"{" selected" if str(k) == str(valeur) else ""}>{C.esc(str(t))}</option>' for k, t in options)
    lab = f'<span class="rdm-cl">{libelle}</span>' if libelle else ""
    return (f'<label class="rdm-c">{lab}<span class="field select"><select data-param="{C.esc(cle)}" aria-label="{C.esc(_brut(libelle) or cle)}">'
            f'{opts}</select>{C.icon("chevron-down")}</span></label>')


def _brut(html_: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", html_ or "")


def bande(valeur: str, unite: str, symbole: str, nom: str, phrase: str, avert: Sequence[str] = ()) -> str:
    """La bande : la valeur choisie en grand, son symbole, son nom, la phrase ; « Pourquoi ? » (conventions, hypothèses,
    domaine de validité) et « Imprimer la feuille »."""
    av = "".join(f'<div class="bib-av">{C.icon("triangle-alert")}<span>{a}</span></div>' for a in avert)
    boutons = (f'<button type="button" class="btn" data-pourquoi="1">{C.icon("lightbulb")}<span>Hypothèses et conventions</span></button>'
               f'<button type="button" class="btn primary" data-imprimer="1">{C.icon("printer")}<span>Imprimer la feuille</span></button>')
    return (f'<div class="bib-bande-v"><b class="bib-val">{valeur}<small>{unite}</small></b>'
            f'<div class="bib-bande-t"><span class="bib-sym">{m(symbole)}</span><span class="bib-nom">{nom}</span>'
            f'<span class="bib-phr">{phrase}</span></div></div>{av}<div class="bib-bande-b">{boutons}</div>')


def volet(titre: str, sous: str, rubriques: Sequence, sources: Sequence[str]) -> str:
    """Le volet « Hypothèses et conventions » : des rubriques (titre, [phrases]) et les sources."""
    corps = ""
    for t, phrases in rubriques:
        corps += f'<h4 class="rdm-vt">{C.esc(t)}</h4><ul class="rdm-vl">' + "".join(f"<li>{p}</li>" for p in phrases) + "</ul>"
    src = "".join(f'<div class="bib-src">{C.icon("file-text")}<span>{s}</span></div>' for s in sources)
    return (f'<div class="bib-pq-h"><div><b>{titre}</b><span>{sous}</span></div>'
            f'<button type="button" class="btn icon ghost" data-fermer-pourquoi="1" aria-label="Fermer">{C.icon("x")}</button></div>'
            f'<div class="bib-pq-b">{corps}<div class="bib-pq-s">{src}</div></div>')


def sources(*textes: str) -> str:
    return "".join(f'<div class="bib-src">{C.icon("file-text")}<span>{t}</span></div>' for t in textes if t)


def entete_impression(titre: str, sous: str, hypotheses: Sequence[str]) -> str:
    """L'en-tête de la feuille IMPRIMÉE (masqué à l'écran) : le titre de l'outil, la date du moteur, les hypothèses."""
    hyp = "".join(f"<li>{h}</li>" for h in hypotheses)
    return (f'<div class="rdm-imp"><div class="rdm-imp-t"><b>Stab Méthodes · {C.esc(titre)}</b><span>{sous}</span>'
            f'<span>{C.esc(TITRE)} {VERSION}</span></div><ul>{hyp}</ul></div>')


def refus(titre: str, raison: str, details: Sequence[str] = ()) -> str:
    """Un cas NON PRIS EN CHARGE, dit clairement : pas de valeur approchée à la place."""
    d = "".join(f"<li>{x}</li>" for x in details)
    return (f'<div class="rdm-refus">{C.icon("triangle-alert")}<div><b>{titre}</b><p>{raison}</p>{("<ul>" + d + "</ul>") if d else ""}</div></div>')


# ================================================================================================ la porte
def _r_poutres(params: Dict, hote: str) -> Dict:
    from api_web import biblio_poutres
    return biblio_poutres.rendre(params, hote)


def _r_plaques(params: Dict, hote: str) -> Dict:
    from api_web import biblio_plaques
    return biblio_plaques.rendre(params, hote)


def _r_sections(params: Dict, hote: str) -> Dict:
    from api_web import biblio_sections
    return biblio_sections.rendre(params, hote)


RENDUS = {"poutres": _r_poutres, "plaques": _r_plaques, "sections": _r_sections}


def consulter(rubrique: Optional[str], params: Optional[Dict], hote: str = "page") -> Dict:
    """La rubrique rendue pour ces paramètres (les absents prennent leur défaut, les illisibles aussi — ``message`` le
    dit) ; une rubrique inconnue ouvre la première."""
    if hote not in HOTES:
        raise ValueError(f"hôte inconnu : {hote!r} (page ou note)")
    k = rubrique if rubrique in RENDUS else RUBRIQUE_DEFAUT
    out = RENDUS[k](params, hote)
    out["rubrique"] = k
    out["bibliotheque"] = BIBLIOTHEQUE
    out.setdefault("fiche", None)
    if rubrique is not None and rubrique != k:
        out["message"] = " ; ".join(x for x in (f"rubrique « {str(rubrique)[:40]} » inconnue : « {NOMS[k]} » ouverte", out.get("message")) if x)
    return out


def menu(actif: str = RUBRIQUE_DEFAUT) -> str:
    """Le menu des rubriques (la page le porte en dur, js/bibliotheque.js allume l'entrée ouverte)."""
    return "".join(f'<button type="button" class="bib-m{" on" if k == actif else ""}" data-rubrique="{k}" aria-pressed="{"true" if k == actif else "false"}">'
                   f'{C.icon(ic)}<span class="bib-mt">{t}</span><span class="bib-ms">{st}</span></button>' for k, t, st, ic in RUBRIQUES)

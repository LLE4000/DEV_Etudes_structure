# -*- coding: utf-8 -*-
"""Les FICHES de la bibliothèque dans une note (docs/site/BIBLIOTHEQUES.md § 2.5) — partagées par les modules de
l'établi compact (poutre, dalle, prédalle).

« Inclure dans la note » copie dans ``etat["fiches"]`` la fiche que la bibliothèque a rendue (api_web/biblio_*.py,
``_fiche``) : la valeur retenue, ses paramètres, ses lignes de justification, ses sources, l'extrait du tableau — tout ce
qu'il faut pour la RELIRE SANS LE MOTEUR (une copie d'un run gelé au sens de R2, comme D-013 : la note garde ce qu'elle
a vu, même si la bibliothèque change ensuite). La clé ``fiches`` est ABSENTE tant que la note n'en a aucune : une note
sans fiche reste identique à l'octet, le gel des modules le prouve.

Ce fichier RELIT strictement (une fiche abîmée est refusée avec un message, jamais réparée en silence), GÈRE la liste
(ajouter, retirer, monter, descendre) et MET EN FORME : le bloc « Justifications (n) » de la colonne de gauche et la vue
« Justification » du centre (``vues``). Il ne calcule rien et ne consulte pas la bibliothèque.

Une fiche : {id, sorte: "fiche" | "extrait", bibliotheque, rubrique, titre, symbole, valeur, texte, unite, phrase, params,
lignes: [{symbole, nom, valeur, texte, unite, regle}], sources, avertissements, extrait: {titre, entetes, lignes, case,
source} | None, date, moteur}. ``texte`` est la valeur telle que la bande l'affichait ; ``valeur`` reste le nombre.
"""
import datetime
import math
from typing import Dict, List, Optional, Sequence

from . import commun as C
from . import etabli_compact as E

SORTES = ("fiche", "extrait")
MAX_FICHES = 24            # fiches par note : au-delà, la note n'est plus une note de calcul
TITRE_MAX = 200
TEXTE_MAX = 600            # une règle, une source, un avertissement, une phrase
LIGNES_MAX = 60
EXTRAIT_MAX = (40, 24)     # lignes, colonnes d'un extrait de tableau
CLES = frozenset({"id", "sorte", "bibliotheque", "rubrique", "titre", "symbole", "valeur", "texte", "unite", "phrase", "params",
                  "lignes", "sources", "avertissements", "extrait", "date", "moteur"})
CLES_LIGNE = frozenset({"symbole", "nom", "valeur", "texte", "unite", "regle"})
CLES_EXTRAIT = frozenset({"titre", "entetes", "lignes", "case", "source"})


# ------------------------------------------------------------------ la relecture stricte
def _texte(x, ou: str, maxi: int = TEXTE_MAX, vide: bool = True) -> str:
    if x is None and vide:
        return ""
    if isinstance(x, bool) or not isinstance(x, (str, int, float)):
        raise ValueError(f"{ou} : un texte est attendu")
    t = str(x).strip()
    if not t and not vide:
        raise ValueError(f"{ou} : vide")
    return t[:maxi]


def _scalaire(x, ou: str):
    """Une valeur : un nombre fini, un booléen, un texte court, ou rien."""
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        if not math.isfinite(x):
            raise ValueError(f"{ou} : nombre non fini")
        return x
    if isinstance(x, str):
        return x.strip()[:TEXTE_MAX]
    raise ValueError(f"{ou} : valeur illisible ({type(x).__name__})")


def _liste(x, ou: str, maxi: int) -> list:
    if x is None:
        return []
    if not isinstance(x, list):
        raise ValueError(f"{ou} : une liste est attendue")
    if len(x) > maxi:
        raise ValueError(f"{ou} : au plus {maxi} éléments")
    return x


def _textes(x, ou: str, maxi: int = 20) -> List[str]:
    return [_texte(t, f"{ou} {i + 1}") for i, t in enumerate(_liste(x, ou, maxi))]


def _params(x, ou: str) -> Dict:
    if x is None:
        return {}
    if not isinstance(x, dict):
        raise ValueError(f"{ou} : un dictionnaire est attendu")
    if len(x) > 40:
        raise ValueError(f"{ou} : au plus 40 paramètres")
    return {_texte(k, ou, 40, vide=False): _scalaire(v, f"{ou} « {str(k)[:40]} »") for k, v in x.items()}


def _ligne(li, ou: str) -> Dict:
    if not isinstance(li, dict):
        raise ValueError(f"{ou} : un dictionnaire est attendu")
    inconnues = set(li) - CLES_LIGNE
    if inconnues:
        raise ValueError(f"{ou} : clés inconnues {sorted(str(k)[:40] for k in inconnues)}")
    return {"symbole": _texte(li.get("symbole"), f"{ou}, symbole", 40), "nom": _texte(li.get("nom"), f"{ou}, nom", TITRE_MAX),
            "valeur": _scalaire(li.get("valeur"), f"{ou}, valeur"), "texte": _texte(li.get("texte"), f"{ou}, texte", 80),
            "unite": _texte(li.get("unite"), f"{ou}, unité", 20), "regle": _texte(li.get("regle"), f"{ou}, règle")}


def _extrait(x, ou: str) -> Optional[Dict]:
    if x is None:
        return None
    if not isinstance(x, dict):
        raise ValueError(f"{ou} : un dictionnaire est attendu")
    inconnues = set(x) - CLES_EXTRAIT
    if inconnues:
        raise ValueError(f"{ou} : clés inconnues {sorted(str(k)[:40] for k in inconnues)}")
    entetes = _textes(x.get("entetes"), f"{ou}, en-têtes", EXTRAIT_MAX[1])
    if not entetes:
        raise ValueError(f"{ou} : sans en-tête")
    lignes = []
    for i, li in enumerate(_liste(x.get("lignes"), f"{ou}, lignes", EXTRAIT_MAX[0])):
        cases = _textes(li, f"{ou}, ligne {i + 1}", EXTRAIT_MAX[1])
        if len(cases) != len(entetes):
            raise ValueError(f"{ou}, ligne {i + 1} : {len(cases)} cases pour {len(entetes)} colonnes")
        lignes.append(cases)
    case = x.get("case")
    if case is not None:
        if (not isinstance(case, (list, tuple)) or len(case) != 2 or any(isinstance(c, bool) or not isinstance(c, int) for c in case)
                or not (0 <= case[0] < len(lignes)) or not (0 <= case[1] < len(entetes))):
            raise ValueError(f"{ou} : case hors du tableau")
        case = [int(case[0]), int(case[1])]
    return {"titre": _texte(x.get("titre"), f"{ou}, titre", TITRE_MAX), "entetes": entetes, "lignes": lignes, "case": case,
            "source": _texte(x.get("source"), f"{ou}, source")}


def relire(f) -> Dict:
    """La fiche relue strictement : chaque clé à sa place, chaque texte borné, l'extrait rectangulaire et sa case
    dedans. Une fiche abîmée lève ValueError avec la raison."""
    if not isinstance(f, dict):
        raise ValueError("fiche illisible : un dictionnaire est attendu")
    inconnues = set(f) - CLES
    if inconnues:
        raise ValueError(f"fiche illisible : clés inconnues {sorted(str(k)[:40] for k in inconnues)}")
    sorte = f.get("sorte", "fiche")
    if sorte not in SORTES:
        raise ValueError(f"fiche illisible : sorte « {str(sorte)[:40]} » inconnue ({', '.join(SORTES)})")
    return {"id": _texte(f.get("id"), "id", 40), "sorte": sorte, "bibliotheque": _texte(f.get("bibliotheque"), "bibliothèque", 40),
            "rubrique": _texte(f.get("rubrique"), "rubrique", 40), "titre": _texte(f.get("titre"), "titre", TITRE_MAX, vide=False),
            "symbole": _texte(f.get("symbole"), "symbole", 40), "valeur": _scalaire(f.get("valeur"), "valeur"),
            "texte": _texte(f.get("texte"), "texte", 80), "unite": _texte(f.get("unite"), "unité", 20),
            "phrase": _texte(f.get("phrase"), "phrase"), "params": _params(f.get("params"), "paramètres"),
            "lignes": [_ligne(li, f"ligne {i + 1}") for i, li in enumerate(_liste(f.get("lignes"), "lignes", LIGNES_MAX))],
            "sources": _textes(f.get("sources"), "source"), "avertissements": _textes(f.get("avertissements"), "avertissement"),
            "extrait": _extrait(f.get("extrait"), "extrait"), "date": _texte(f.get("date"), "date", 20),
            "moteur": _texte(f.get("moteur"), "moteur", 80)}


def relire_toutes(x) -> List[Dict]:
    """Les fiches d'une note relue (``normaliser`` des modules) : une liste, chaque fiche relue strictement."""
    if not isinstance(x, list):
        raise ValueError(f"note illisible : « fiches » doit être une liste (reçu {type(x).__name__})")
    if len(x) > MAX_FICHES:
        raise ValueError(f"note illisible : au plus {MAX_FICHES} fiches")
    return [relire(f) for f in x]


# ------------------------------------------------------------------ la liste de la note
def gerer(etat: Dict, quoi: str, fiche=None, index=None) -> Optional[str]:
    """« ajouter » (``fiche`` : ce que la bibliothèque a rendu, complété par l'écran de la sorte, de la rubrique et de la
    bibliothèque ; la date du jour si elle manque), « retirer », « monter », « descendre » (``index``). Rend un message
    quand la demande est refusée — l'état reste alors intact."""
    fiches = list(etat.get("fiches") or [])
    if quoi == "ajouter":
        if len(fiches) >= MAX_FICHES:
            return f"Au plus {MAX_FICHES} fiches dans une note"
        try:
            f = relire(fiche)
        except ValueError as e:
            return f"Fiche refusée : {e}"
        if not f["date"]:
            f["date"] = datetime.date.today().isoformat()
        pris = {x["id"] for x in fiches}
        if not f["id"] or f["id"] in pris:
            n = len(fiches) + 1
            while f"j{n}" in pris:
                n += 1
            f["id"] = f"j{n}"
        fiches.append(f)
    else:
        try:
            i = int(index)
        except (TypeError, ValueError):
            return "Fiche inconnue"
        if isinstance(index, bool) or not 0 <= i < len(fiches):
            return "Fiche inconnue"
        if quoi == "retirer":
            del fiches[i]
        elif quoi == "monter":
            if i > 0:
                fiches[i - 1], fiches[i] = fiches[i], fiches[i - 1]
        elif quoi == "descendre":
            if i < len(fiches) - 1:
                fiches[i + 1], fiches[i] = fiches[i], fiches[i + 1]
        else:
            return f"Action inconnue : « {str(quoi)[:40]} »"
    if fiches:
        etat["fiches"] = fiches
    else:
        etat.pop("fiches", None)
    return None


# ------------------------------------------------------------------ la mise en forme
def valeur_html(f: Dict) -> str:
    """« c_nom = 35 mm », le symbole composé, la valeur telle que la bande l'affichait."""
    sym = C.symb_math(f["symbole"]) + " = " if f["symbole"] else ""
    return f'{sym}{C.esc(f["texte"] if f["texte"] else C.nb(f["valeur"]) if isinstance(f["valeur"], (int, float)) else str(f["valeur"] or "—"))}{(C.NBSP + C.esc(f["unite"])) if f["unite"] else ""}'


def bloc(fiches: Sequence[Dict]) -> str:
    """Le bloc « Justifications (n) » de la colonne de gauche, sous l'arborescence : une ligne par fiche (son titre, sa
    valeur), un clic ouvre sa vue au centre ; le menu « ⋯ » la monte, la descend, la retire. Rien quand la note n'a
    aucune fiche : le bouton « Bibliothèque » du ruban est l'entrée."""
    if not fiches:
        return ""
    out = ""
    n = len(fiches)
    for i, f in enumerate(fiches):
        actions = (E.action(f'data-fiche-monter="{i}"', "arrow-up", "Monter", False, i > 0)
                   + E.action(f'data-fiche-descendre="{i}"', "arrow-down", "Descendre", False, i < n - 1)
                   + E.action(f'data-fiche-retirer="{i}"', "trash-2", "Retirer de la note", True))
        out += (f'<div class="et-fi" data-fiche="{i}"><button type="button" class="et-fi-t" data-vue-fiche="fiche-{i}" title="Voir la fiche au centre">'
                f'{C.icon("book-open")}<span class="t"><span class="nom">{C.esc(f["titre"])}</span><span class="dim">{valeur_html(f)}</span></span></button>'
                f'{E.menu(actions, "Actions de la fiche")}</div>')
    return f'<div class="et-fiches"><div class="et-arbre-h"><span>Justifications</span><span class="n">{n}</span></div>{out}</div>'


def _ligne_html(li: Dict) -> str:
    maj = li["nom"].isupper()
    v = li["texte"] if li["texte"] else (C.nb(li["valeur"]) if isinstance(li["valeur"], (int, float)) and not isinstance(li["valeur"], bool)
                                         else ("oui" if li["valeur"] is True else "non" if li["valeur"] is False else str(li["valeur"] or "—")))
    u = f' <span class="u">{C.esc(li["unite"])}</span>' if li["unite"] else ""
    cls = ' class="maj"' if maj else ""
    return (f'<tr{cls}><td class="s">{C.symb_math(li["symbole"])}</td><td class="nm">{C.esc(li["nom"].capitalize() if maj else li["nom"])}</td>'
            f'<td class="n">{C.esc(v)}{u}</td><td class="rg">{C.esc(li["regle"])}</td></tr>')


def _extrait_html(x: Dict) -> str:
    case = x["case"] or (-1, -1)
    th = "".join(f'<th class="nm">{C.symb_math(e)}</th>' if j == 0 else f'<th>{C.symb_math(e)}</th>' for j, e in enumerate(x["entetes"]))
    corps = ""
    for i, li in enumerate(x["lignes"]):
        on = case[0] == i
        cases = ""
        for j, c in enumerate(li):
            if j == 0:
                cases += f'<th class="nm">{C.esc(c)}</th>'
            else:
                cls = "cur" if (on and case[1] == j) else ("col" if case[1] == j else "")
                cases += f'<td class="{cls}">{C.esc(c)}</td>'
        corps += f'<tr class="on">{cases}</tr>' if on else f'<tr>{cases}</tr>'
    src = f'<div class="src">{C.icon("file-text")}<span>{C.esc(x["source"])}</span></div>' if x["source"] else ""
    return (f'<div class="et-fiche-x"><div class="k">{C.esc(x["titre"])}</div><div class="et-fiche-xc"><table class="et-fiche-xt"><thead><tr>{th}</tr></thead>'
            f'<tbody>{corps}</tbody></table></div>{src}</div>')


def date_fr(iso: str) -> str:
    """« 2026-11-09 » (l'état garde la date ISO, qui se trie) → « 09/11/2026 » (l'écran et la note l'écrivent ainsi)."""
    s = str(iso or "")
    return f"{s[8:10]}/{s[5:7]}/{s[:4]}" if len(s) == 10 and s[4] == "-" and s[7] == "-" else s


def vue_html(f: Dict, i: int) -> str:
    """La vue « Justification » : la fiche telle qu'elle a été incluse — titre, valeur en grand, phrase des paramètres,
    lignes de justification, avertissements, extrait du tableau (la case marquée), sources, date."""
    lignes = "".join(_ligne_html(li) for li in f["lignes"])
    av = "".join(f'<div class="et-fiche-av">{C.icon("triangle-alert")}<span>{C.esc(a)}</span></div>' for a in f["avertissements"])
    src = "".join(f'<span class="src">{C.icon("file-text")}<span>{C.esc(s)}</span></span>' for s in f["sources"])
    ou = " · ".join(t for t in (f["bibliotheque"] and f"Bibliothèque {C.esc(f['bibliotheque'])}", f["rubrique"] and C.esc(f["rubrique"])) if t)
    pied = " · ".join(t for t in (f["date"] and C.esc(date_fr(f["date"])), f["moteur"] and C.esc(f["moteur"])) if t)
    table = f'<table class="et-fiche-t"><tbody>{lignes}</tbody></table>' if lignes else ""
    morceaux = [av, table] if f["sorte"] == "fiche" else [av]
    if f["extrait"]:
        morceaux.insert(0 if f["sorte"] == "extrait" else len(morceaux), _extrait_html(f["extrait"]))
    if f["sorte"] == "extrait" and table:
        morceaux.append(table)
    ph = f'<span class="ph">{C.esc(f["phrase"])}</span>' if f["phrase"] else ""
    k = f"Justification J{i + 1}" + (f" · {ou}" if ou else "")
    grand = C.esc(f["texte"] or (C.nb(f["valeur"]) if isinstance(f["valeur"], (int, float)) and not isinstance(f["valeur"], bool) else "—"))
    return (f'<div class="et-fiche" data-fiche-vue="{i}"><div class="et-fiche-h"><div class="et-fiche-ht"><span class="k">{k}</span>'
            f'<b>{C.esc(f["titre"])}</b>{ph}</div>'
            f'<div class="et-fiche-v"><b>{grand}<small>{C.esc(f["unite"])}</small></b><span>{C.symb_math(f["symbole"])}</span></div></div>'
            + "".join(morceaux)
            + f'<div class="et-fiche-p"><span class="et-fiche-s">{src}</span><span class="et-fiche-d">{pied}</span></div></div>')


def vues(fiches: Sequence[Dict]) -> List[Dict]:
    """Une vue du centre par fiche : clé « fiche-<i> » (jamais « s<i> » : js/compact.js lit « s0 », « c1 »… comme la
    section d'une vue), onglet « J1 », le HTML de la fiche (un bloc, pas un dessin)."""
    return [{"cle": f"fiche-{i}", "titre": f["titre"], "court": f"J{i + 1}", "svg": vue_html(f, i),
             "legende": "fiche de la bibliothèque, copiée dans la note telle qu'elle a été consultée", "echelle": False}
            for i, f in enumerate(fiches)]

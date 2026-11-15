# -*- coding: utf-8 -*-
"""Le module « Générateur SAF » du site — premier outil : FONDATION DE GRUE. On dessine la dalle, on choisit le châssis
(la bibliothèque des châssis de grue, ou une saisie) et on le place, on donne les réactions des quatre pieds : le module
fait tourner les charges dans toutes les positions, estime tout de suite la contrainte de contact et le tassement pour
plusieurs raideurs de sol, place les bandes d'intégration, et exporte les fichiers SAF (un par k) avec la note.

Rien n'est calculé ici : le moteur est ``saf/fondation_grue`` (le projet et son modèle, la plaque sur sol élastique en
compression seule, « Proposer », l'écriture SAF) ; ce fichier tient l'ÉTAT de la note, écrit les saisies et assemble
l'écran — la saisie (``saf_saisie``), le plan (``dessin_saf``), les vérifications (``saf_resultat``), la note
(``note_saf``).

Une note : ``{"version": 1, "fondations": [projet, …], "actif": 0}`` — un projet est celui de
``saf.fondation_grue.projet`` (``nouveau``, ``lire``). L'écran est l'établi COMPACT : à gauche la ligne de la fondation
ouverte et ses rubriques (Projet, Dalle, Sol, Grue, Réactions, Bandes, Limite, puis Options, repliée) ; au centre le PLAN,
qui se manipule (js/plan.js : le châssis se déplace et se tourne, les cotes se tapent, le contour se dessine) ; à droite
les vérifications rapides, mises à jour à chaque modification.

LE CARNET (``CARNET``) : les GRUES ENREGISTRÉES servent d'une note à l'autre. Elles ne sont donc pas dans la note : l'écran
les garde (js/etabli.js, dans ce navigateur) et les joint à chaque requête ; le routeur les pose dans ``etat["carnet"]``
le temps de la requête et les rend dans la réponse (``carnet``) — jamais dans l'état enregistré.
"""
import base64
import copy
import datetime
import io
import json
import zipfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from saf.fondation_grue import chassis as CH
from saf.fondation_grue import ecriture as EC
from saf.fondation_grue import efforts as EF
from saf.fondation_grue import geometrie as G
from saf.fondation_grue import projet as PR
from saf.fondation_grue import proposition as PP
from saf.fondation_grue import verification as V

from . import commun as C
from . import dessin_saf as DS
from . import etabli_compact as E
from . import saf_resultat as SR
from . import saf_saisie as SS

MODULE = "saf"
PAGE = "generateur-saf"
CARNET = True                    # les grues enregistrées voyagent avec la requête (voir l'en-tête du fichier)
TITRE = "Fondation de grue"
GENRE = E.Genre("Fondations", "la fondation", "cette fondation", "une fondation")
CARNET_MAX = 40                  # grues enregistrées, au plus
HORS_CALCUL = ("dmin", "repere")  # ce qui, de la grue, ne change pas le calcul (la distance mini demandée, le repère des cotes)
TYPE_ZIP = "application/zip"


# ------------------------------------------------------------------ l'état
def nouvel_etat() -> Dict:
    return {"version": 1, "fondations": [PR.nouveau("F1")], "actif": 0}


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu par le moteur : chaque projet contrôlé, la fondation ouverte
    valide. Le carnet n'est jamais dans la note."""
    fs = [PR.lire(f) for f in (etat or {}).get("fondations") or []] or [PR.nouveau("F1")]
    try:
        actif = int((etat or {}).get("actif") or 0)
    except (TypeError, ValueError):
        actif = 0
    return {"version": 1, "fondations": fs, "actif": min(max(0, actif), len(fs) - 1)}


def _f(etat: Dict) -> Dict:
    return etat["fondations"][etat["actif"]]


def _indice(x, n: int) -> int:
    try:
        i = int(x)
    except (TypeError, ValueError):
        raise ValueError("rang illisible")
    if not 0 <= i < n:
        raise ValueError("rang hors de la liste")
    return i


def _nom_libre(fs: List[Dict]) -> str:
    pris = {f.get("nom") for f in fs}
    k = len(fs) + 1
    while f"F{k}" in pris:
        k += 1
    return f"F{k}"


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les fondations de la note : en ouvrir une, en ajouter, en copier une (la copie s'insère après), en supprimer une
    (jamais la dernière)."""
    fs = etat["fondations"]

    def rang(x):
        try:
            return min(max(0, int(x)), len(fs) - 1)
        except (TypeError, ValueError):
            return 0
    if action == "activer":
        etat["actif"] = rang(index)
    elif action == "ajouter":
        fs.append(PR.nouveau(_nom_libre(fs)))
        etat["actif"] = len(fs) - 1
    elif action == "dupliquer":
        i = etat["actif"] if index is None else rang(index)
        c = copy.deepcopy(fs[i])
        c["nom"] = _nom_libre(fs)
        fs.insert(i + 1, c)
        etat["actif"] = i + 1
    elif action == "supprimer" and len(fs) > 1:
        i = rang(index)
        del fs[i]
        if etat["actif"] > i or etat["actif"] >= len(fs):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
def _nombre(valeur, quoi: str) -> float:
    x = valeur if isinstance(valeur, (int, float)) and not isinstance(valeur, bool) else C.lire_nombre(valeur)
    if x is None:
        raise ValueError(f"{quoi} : nombre illisible (« {str(valeur)[:20]} »)")
    return float(x)


def _bool(valeur) -> bool:
    b = C.lire_bool(valeur)
    if b is None:
        raise ValueError("oui ou non attendu")
    return b


def _poser_contour(el: Dict, pts: Sequence) -> None:
    """Le contour devient ``pts`` : la grue reste OÙ ELLE EST (ses cotes au centre de la dalle suivent), puis le contour
    est ramené à l'origine de sa boîte, dans le sens trigonométrique."""
    d, g = el["dalle"], el["grue"]
    anciens = [tuple(p) for p in d["sommets"]]
    pts = [(round(float(x), 3), round(float(y), 3)) for x, y in pts]
    if len(pts) > PR.SOMMETS_MAX:
        raise ValueError(f"le contour a {PR.SOMMETS_MAX} sommets au plus")
    if len(anciens) >= 3:                                         # où est la grue, dans le repère des anciens sommets
        x0, y0, x1, y1 = G.boite(anciens)
        gx, gy = (x0 + x1) / 2 + g["ex"], (y0 + y1) / 2 + g["ey"]
    else:
        gx = gy = None
    x0, y0, x1, y1 = G.boite(pts)
    pts = G.sens_trigo(pts)
    d["sommets"] = [[round(x - x0, 3), round(y - y0, 3)] for x, y in pts]
    g["repere"] = None                                            # les bords ont changé : les cotes reviennent aux axes
    if gx is not None:
        g["ex"], g["ey"] = round(gx - (x0 + x1) / 2, 3), round(gy - (y0 + y1) / 2, 3)


def _inserer_sommet(pts: List, x: float, y: float, apres: Optional[int]) -> List:
    """Les sommets, avec (x, y) de plus : après le sommet ``apres``, ou sur le bord le plus proche."""
    if apres is None:
        apres = min(range(len(pts)), key=lambda i: G.distance_point_segment((x, y), pts[i], pts[(i + 1) % len(pts)]))
    return pts[:apres + 1] + [(x, y)] + pts[apres + 1:]


def _ecrire(etat: Dict, el: Dict, ch: List[str], valeur) -> Optional[str]:
    """Écrit une saisie dans le projet ``el`` (une copie, que ``modifier`` fait relire) ; lève ValueError pour un refus."""
    d, g, o = el["dalle"], el["grue"], el["options"]
    t = ch[0]
    if ch == ["nom"]:
        nom = " ".join(str(valeur or "").split())
        if not nom:
            raise ValueError("un nom ne peut pas être vide")
        if len(nom) > PR.NOM_MAX:
            raise ValueError(f"le nom : {PR.NOM_MAX} caractères au plus ({len(nom)} saisis) — il titre la note et le fichier")
        el["nom"] = nom
    elif t == "projet" and len(ch) == 2 and ch[1] in el["projet"]:
        el["projet"][ch[1]] = str(valeur or "")
    elif t == "dalle":
        if ch[1] == "forme":
            if valeur == "contour" and len(d["sommets"]) < 3:    # le contour part du rectangle
                d["sommets"] = [[0.0, 0.0], [d["Lx"], 0.0], [d["Lx"], d["Ly"]], [0.0, d["Ly"]]]
            d["forme"] = valeur
            g["repere"] = None
        elif ch[1] in ("Lx", "Ly", "h"):
            d[ch[1]] = round(_nombre(valeur, {"Lx": "L_x", "Ly": "L_y", "h": "l'épaisseur"}[ch[1]]), 3)
        elif ch[1] == "beton":
            d["beton"] = valeur
        elif ch[1] == "sommets" and len(ch) == 3 and ch[2] == "+":
            pts = [tuple(p) for p in d["sommets"]]
            a, b = pts[-1], pts[0]
            _poser_contour(el, pts + [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)])
        elif ch[1] == "sommets" and len(ch) == 4 and ch[3] == "-":
            i = _indice(ch[2], len(d["sommets"]))
            if len(d["sommets"]) <= 3:
                raise ValueError("le contour garde au moins 3 sommets")
            _poser_contour(el, [tuple(p) for j, p in enumerate(d["sommets"]) if j != i])
        elif ch[1] == "sommets" and len(ch) == 4 and ch[3] in ("0", "1"):
            i = _indice(ch[2], len(d["sommets"]))
            pts = [list(p) for p in d["sommets"]]
            pts[i][int(ch[3])] = _nombre(valeur, "le sommet")
            _poser_contour(el, pts)
        elif ch[1] == "sommet" and len(ch) == 3 and ch[2] == "+":
            if not isinstance(valeur, dict):
                raise ValueError("sommet illisible")
            pts = [tuple(p) for p in d["sommets"]]
            apres = None if valeur.get("apres") is None else _indice(valeur.get("apres"), len(pts))
            _poser_contour(el, _inserer_sommet(pts, _nombre(valeur.get("x"), "x"), _nombre(valeur.get("y"), "y"), apres))
        elif ch[1] == "sommet" and len(ch) == 3:
            i = _indice(ch[2], len(d["sommets"]))
            if not isinstance(valeur, (list, tuple)) or len(valeur) != 2:
                raise ValueError("sommet illisible")
            pts = [list(p) for p in d["sommets"]]
            pts[i] = [_nombre(valeur[0], "x"), _nombre(valeur[1], "y")]
            _poser_contour(el, pts)
        else:
            raise ValueError("champ inconnu")
    elif t == "sol":
        ks = el["sol"]["k"]
        if ch[1:] == ["k", "+"]:
            if len(ks) >= PR.K_MAX_VALEURS:
                raise ValueError(f"{PR.K_MAX_VALEURS} valeurs de k au plus")
            ks.append({"k": _k_suivant([x["k"] for x in ks]), "scia": False})
        elif ch[1] == "k" and len(ch) == 4 and ch[3] == "-":
            if len(ks) <= 1:
                raise ValueError("il faut au moins une valeur de k")
            del ks[_indice(ch[2], len(ks))]
        elif ch[1] == "k" and len(ch) == 4 and ch[3] == "k":
            ks[_indice(ch[2], len(ks))]["k"] = _nombre(valeur, "k")
        elif ch[1] == "k" and len(ch) == 4 and ch[3] == "scia":
            ks[_indice(ch[2], len(ks))]["scia"] = _bool(valeur)
        elif ch[1:] == ["reprise"]:
            return _reprendre_sol(el, valeur)
        elif ch[1:] == ["reprise", "-"]:
            el["sol"]["reprise"] = None
        else:
            raise ValueError("champ inconnu")
    elif t == "grue":
        if ch[1] == "chassis":
            _poser_chassis(el, valeur)
        elif ch[1] in ("a", "b", "ex", "ey", "beta", "dmin"):
            noms = {"a": "l'entraxe a", "b": "l'entraxe b", "ex": "e_x", "ey": "e_y", "beta": "l'angle β", "dmin": "la distance mini"}
            g[ch[1]] = round(_nombre(valeur, noms[ch[1]]), 4)
        elif ch[1] == "position":
            if not isinstance(valeur, (list, tuple)) or len(valeur) != 2:
                raise ValueError("position illisible")
            g["ex"], g["ey"] = round(_nombre(valeur[0], "e_x"), 3), round(_nombre(valeur[1], "e_y"), 3)
        elif ch[1] == "repere":
            # un bord touché devient le repère des cotes ; le même, touché de nouveau, rend les axes de la dalle
            i = None if valeur in (None, "") else _indice(valeur, 4 if d["forme"] == "rectangle" else len(d["sommets"]))
            g["repere"] = None if i == g.get("repere") else i
        elif ch[1] == "cote" and len(ch) == 3 and ch[2] in ("perp", "long"):
            x = _nombre(valeur, "la cote")
            g["ex"], g["ey"] = (round(v, 3) for v in PR.position_au_bord(el, PR.modele(el), **{"perp" if ch[2] == "perp" else "long_": x}))
        elif ch[1] == "appuis":
            g["appuis"] = valeur
        elif ch[1] == "rect" and len(ch) == 3 and ch[2] in ("a", "b"):
            g["rect"][ch[2]] = round(_nombre(valeur, "la plaque"), 3)
        elif ch[1:] == ["rect", "angle"]:
            g["rect"]["angle"] = {"0": 0, "45": 45}.get(str(valeur))
        elif ch[1:] == ["plots", "type"]:
            # (les dimensions par défaut de la sorte choisie — préfabriqué 2,50 × 2,50 × 0,35 m, coulé 2,00 × 2,00 × 0,35 m —
            # remplacent celles d'une AUTRE sorte ; des dimensions saisies à la main sont gardées)
            p_ = g["plots"]
            if valeur in PR.PLOTS_DEFAUT and (p_["a"], p_["b"], p_["h"]) in [v for k, v in PR.PLOTS_DEFAUT.items() if k != valeur]:
                p_["a"], p_["b"], p_["h"] = PR.PLOTS_DEFAUT[valeur]
            p_["type"] = valeur
        elif ch[1] == "plots" and len(ch) == 3 and ch[2] in ("a", "b", "h"):
            g["plots"][ch[2]] = round(_nombre(valeur, "le plot"), 3)
        else:
            raise ValueError("champ inconnu")
    elif t == "reactions":
        jeux = el["reactions"]
        if ch[1:] == ["+"]:
            if len(jeux) >= PR.JEUX_MAX:
                raise ValueError(f"{PR.JEUX_MAX} jeux de réactions au plus")
            jeux.append({"nom": "", "P": [None] * 4})
        elif len(ch) == 3 and ch[2] == "-":
            i = _indice(ch[1], len(jeux))
            if len(jeux) > 1:
                del jeux[i]
            else:                                                # le dernier jeu se vide, il ne disparaît pas
                jeux[0] = {"nom": "", "P": [None] * 4}
        elif len(ch) == 3 and ch[2] == "nom":
            jeux[_indice(ch[1], len(jeux))]["nom"] = str(valeur or "")
        elif len(ch) == 4 and ch[2] == "P":
            vide = valeur is None or str(valeur).strip() == ""
            jeux[_indice(ch[1], len(jeux))]["P"][_indice(ch[3], 4)] = None if vide else _nombre(valeur, "la réaction")
        else:
            raise ValueError("champ inconnu")
    elif ch == ["bandes", "largeur"]:
        el["bandes"]["largeur"] = _nombre(valeur, "la largeur")
    elif ch == ["limite"]:
        # (saisie dans l'unité choisie — lot 4, § 2.4 — et gardée en kg/cm²)
        el["limite"] = _nombre(valeur, "la limite") * SS.unite(el)[1] / V.KG_CM2
    elif t == "armatures" and len(ch) == 2 and ch[1] in el["armatures"]:
        if ch[1] == "enrobage":
            el["armatures"]["enrobage"] = round(_nombre(valeur, "l'enrobage") / 100.0, 4)      # (saisi en cm)
        else:
            el["armatures"][ch[1]] = str(valeur)
    elif t == "options" and len(ch) == 2 and ch[1] in o:
        if ch[1] in ("aretes", "nonlinear", "torsion"):
            o[ch[1]] = _bool(valeur)
        elif ch[1] in ("rond", "unite"):
            o[ch[1]] = valeur
        elif ch[1] == "anneau":
            o["anneau"] = _nombre(valeur, "l'anneau")
        else:
            o[ch[1]] = _nombre(valeur, ch[1])
    elif t == "vue" and len(ch) == 2 and ch[1] in el["vue"]:
        if ch[1] == "carte":
            if valeur == "m":
                valeur = el["vue"]["carte"] if el["vue"]["carte"] in PR.CARTES_MOMENTS else PR.CARTES_MOMENTS[0]
            if valeur not in PR.CARTES:
                raise ValueError("carte : contrainte, tassement, moments ou sans carte")
            el["vue"]["carte"] = valeur
        elif ch[1] == "onglet":
            el["vue"][ch[1]] = valeur
        elif ch[1] == "cas":
            el["vue"]["cas"] = str(valeur) if valeur not in (None, "") else None
        else:
            el["vue"]["k"] = None if valeur in (None, "") else _nombre(valeur, "k")
    elif ch == ["proposition", "garder"]:
        lignes = _PROPOSITIONS.get(_signature_proposition(el))
        if not lignes or any(h not in lignes for h in PR.epaisseurs(el)):
            raise ValueError("la proposition n'est pas complète : relancer « Proposer »")
        el["proposition"] = {"signature": _signature_proposition(el), "lignes": [lignes[h] for h in PR.epaisseurs(el)]}
    elif ch == ["proposition", "-"]:
        el["proposition"] = None
    elif ch == ["proposition", "appliquer"]:
        p = el.get("proposition") or {}
        li = (p.get("lignes") or [])[_indice(valeur, len(p.get("lignes") or []))]
        if li.get("d") is None:
            raise ValueError("cette épaisseur ne tient pas")
        d["h"] = li["h"]
        if li.get("sommets"):
            _poser_contour(el, li["sommets"])
        else:
            d["Lx"], d["Ly"] = li["Lx"], li["Ly"]
    elif t == "carnet":
        return _carnet(etat, el, ch[1:], valeur)
    else:
        raise ValueError("champ inconnu")
    return None


def _k_suivant(ks: List[float]) -> float:
    """La valeur de k qu'on ajoute : la suite du bureau (2, 5, 10, 20, 30, 50…), la première qui n'y est pas."""
    for k in (2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0):
        if k not in ks:
            return k
    return max(ks) + 50.0


def _poser_chassis(el: Dict, ident) -> None:
    """Le châssis choisi : un id de la bibliothèque (appuis réels par défaut, orientation de sa platine), ou la saisie."""
    g = el["grue"]
    if ident in (None, "", CH.SAISIE):
        g["chassis"] = CH.SAISIE
        g["rect"]["angle"] = None                                # (relu : 0° pour un châssis saisi)
    elif CH.existe(ident):
        g["chassis"], g["appuis"] = ident, "reels"
        g["rect"]["angle"] = None
    else:
        raise ValueError("ce châssis n'est pas dans la bibliothèque")


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la fondation ouverte ; rend un message si elle est refusée (la note n'a alors pas changé),
    une information parfois, None sinon. La saisie est écrite sur une copie, que le moteur RELIT (``projet.lire``) : une
    valeur hors de ses bornes, deux k égaux, un contour qui se recoupe sont refusés avec leur raison."""
    f = _f(etat)
    essai = copy.deepcopy(f)
    try:
        message = _ecrire(etat, essai, [str(c) for c in chemin], valeur)
        relu = PR.lire(essai)
    except ValueError as e:
        return SS.phrase(str(e))
    # une proposition calculée pour d'autres données n'en est plus une
    if relu.get("proposition") and relu["proposition"].get("signature") != _signature_proposition(relu):
        relu["proposition"] = None
    etat["fondations"][etat["actif"]] = relu
    return message


# ------------------------------------------------------------------ le châssis : la liste, le sélecteur
def _libelle_chassis(c: Dict) -> str:
    return c["nom"] if c["constructeur"] and c["constructeur"].lower() in c["nom"].lower() else f"{c['constructeur']} {c['nom']}".strip()


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le clic dans le sélecteur des châssis : « Saisie », ou le châssis de la bibliothèque qui porte ce libellé."""
    if role != "chassis":
        return
    f = _f(etat)
    if nom == SS.SAISIE_TXT:
        _poser_chassis(f, CH.SAISIE)
    else:
        ident = next((c["id"] for c in CH.liste() if _libelle_chassis(c) == nom), None)
        if ident is None:
            raise ValueError("ce châssis n'est pas dans la bibliothèque")
        _poser_chassis(f, ident)
    etat["fondations"][etat["actif"]] = PR.lire(f)


def selecteur(etat: Dict, role: str, familles: Optional[Sequence[str]] = None, froid: bool = True) -> str:
    """« chassis » : la liste des châssis de la bibliothèque, par constructeur (les puces), avec la recherche ;
    « proposer:<h> » : la ligne de « Proposer » pour l'épaisseur h — une LECTURE, la note ne change pas."""
    f = _f(etat)
    if role.startswith("proposer:"):
        return _proposer(f, role.split(":", 1)[1])
    if role != "chassis":
        raise ValueError(f"sélecteur inconnu : {role}")
    tous = CH.liste()
    constructeurs = list(dict.fromkeys(c["constructeur"] for c in tous))
    retenus = [k for k in constructeurs if not familles or k in familles]
    cur = f["grue"]["chassis"]
    rows = [{"nom": SS.SAISIE_TXT, "col": f"{C.fr(f['grue']['a'], 2)} × {C.fr(f['grue']['b'], 2)}{C.NBSP}m", "taux": None, "niveau": "c",
             "cur": cur == CH.SAISIE, "opt": False, "prop": "Un châssis hors de la bibliothèque : ses entraxes et ses plaques se saisissent."}]
    for c in tous:
        if c["constructeur"] not in retenus:
            continue
        ch = CH.lire({"chassis": c["id"], "appuis": "reels"})
        prop = (f"{C.esc(c['constructeur'])} · entraxes {C.fr(ch['a'], 2)} × {C.fr(ch['b'], 2)}{C.NBSP}m · appuis : "
                + (f"{C.esc(ch['dims'])} ({C.esc(ch['forme'])}), {C.fr(ch['aire'], 4)}{C.NBSP}m²" if not ch["impose"] else "platine non indiquée")
                + (f" · {C.esc(ch['note_fondation'])}" if ch["note_fondation"] else ""))
        rows.append({"nom": _libelle_chassis(c), "col": f"{C.fr(ch['a'], 2)} × {C.fr(ch['b'], 2)}{C.NBSP}m", "taux": None, "niveau": "c",
                     "cur": cur == c["id"], "opt": False, "prop": prop, "etat": "sans platine" if ch["impose"] else ""})
    filtres = ('<input type="search" class="pk-cherche" data-cherche="1" placeholder="Chercher un châssis" aria-label="Chercher un châssis" spellcheck="false">'
               + "".join(f'<button type="button" class="fchip{" on" if k in retenus else ""}" data-famille="{C.esc(k)}">{C.esc(k)}</button>'
                         for k in constructeurs))
    i_c = next((i for i, r in enumerate(rows) if r["cur"]), 0)
    return C.selecteur("chassis", "Châssis", "la bibliothèque des châssis de grue · un clic retient le châssis", filtres, rows, max(0, i_c - 4),
                       "entraxes a × b", etoile="", pastille="")


# ------------------------------------------------------------------ le calcul (gardé d'une requête à l'autre)
_CALCULS: "OrderedDict[str, Dict]" = OrderedDict()
CALCULS_MAX = 12


def _signature_calcul(el: Dict) -> str:
    g = {k: v for k, v in el["grue"].items() if k not in HORS_CALCUL}
    o = el["options"]
    return json.dumps([el["dalle"], [x["k"] for x in el["sol"]["k"]], g, [j["P"] for j in el["reactions"]],
                       [o["maille"], o["c2x"], o["c2y"], o["rond"]]], sort_keys=True)


def calculer(el: Dict) -> Dict:
    """Le calcul d'une fondation : {"modele", "par_k" (le résultat de chaque k de la liste, ou None), "refus" (pourquoi
    le calcul n'a pas eu lieu, ou None)}. Le résultat des k est gardé : changer la limite, les bandes, la vue ou le nom
    ne recalcule rien."""
    m = PR.modele(el)
    ctl = m["controles"]
    sig = _signature_calcul(el)
    if sig in _CALCULS:
        _CALCULS.move_to_end(sig)
        return {"modele": m, **_CALCULS[sig]}
    par_k, refus = None, None
    if not all(ctl["plaques_dedans"]):
        refus = "Une surface d'appui sort de la dalle."
    elif m["plots"] and not all(ctl["plots_dedans"]):
        refus = "Un plot sort de la dalle."
    else:
        o = el["options"]
        try:
            par_k = V.par_k(m, [x["k"] for x in el["sol"]["k"]], (o["c2x"], o["c2y"]))
        except ValueError as e:
            refus = SS.phrase(str(e))
    _CALCULS[sig] = {"par_k": par_k, "refus": refus}
    while len(_CALCULS) > CALCULS_MAX:
        _CALCULS.popitem(last=False)
    return {"modele": m, "par_k": par_k, "refus": refus}


# ------------------------------------------------------------------ les efforts ELS (lot 4, § 4)
_EFFORTS: "OrderedDict[str, Dict]" = OrderedDict()


def efforts(el: Dict, calc: Dict, ks: Optional[List[Dict]] = None, cas: Optional[str] = None) -> Optional[Dict]:
    """Les efforts ELS de la fondation (``saf.fondation_grue.efforts.enveloppe``), LUS dans les solutions déjà calculées —
    aucun calcul en plus (§ 4.1) : par défaut l'enveloppe de tous les cas et de tous les k cochés « SCIA » (ce que le
    ferraillage emploie) ; ``ks`` et ``cas`` la restreignent (la carte « Moments » d'un k ou d'un cas choisis). Le
    résultat est gardé : changer la vue, la limite ou le nom ne le refait pas. None sans calcul.
    (d, qui borne la lecture de V à d, se donne plus tard — ``efforts.v_a_d`` relit V(s) — : la valeur passée ici,
    h − 6 cm, ne sert qu'à remplir la clé « V_d ».)"""
    if calc["par_k"] is None:
        return None
    if ks is None:
        scia = set(PR.k_scia(el))
        ks = [r for r in calc["par_k"] if r["k"] in scia] or list(calc["par_k"])
    if not ks:
        return None
    torsion = bool(el["options"]["torsion"])
    cle = json.dumps([_signature_calcul(el), [r["k"] for r in ks], cas, torsion, el["bandes"]["largeur"]])
    if cle in _EFFORTS:
        _EFFORTS.move_to_end(cle)
        return _EFFORTS[cle]
    if cas is not None:
        ks = [{"k": r["k"], "dalle": r["dalle"], "deplacements": [x for c, x in zip(r["cas"], r["deplacements"]) if c["nom"] == cas]}
              for r in ks]
    ef = EF.enveloppe(calc["modele"], ks, max(el["dalle"]["h"] - 0.06, 0.01), torsion)
    _EFFORTS[cle] = ef
    while len(_EFFORTS) > CALCULS_MAX:
        _EFFORTS.popitem(last=False)
    return ef


# ------------------------------------------------------------------ « Proposer »
_PROPOSITIONS: "OrderedDict[str, Dict[float, Dict]]" = OrderedDict()


def _signature_proposition(el: Dict) -> str:
    """Ce dont une proposition dépend : tout, SAUF les dimensions de la dalle (elle les propose) et ce qui ne touche pas
    au calcul."""
    d = el["dalle"]
    o = el["options"]
    return json.dumps([d["forme"], d["beton"], PR.k_scia(el), {k: v for k, v in el["grue"].items() if k not in HORS_CALCUL},
                       [j["P"] for j in el["reactions"]], el["limite"], [o["c2x"], o["c2y"], o["rond"], o["h_min"], o["h_max"], o["h_pas"]]],
                      sort_keys=True)


def _proposer(el: Dict, h_txt: str) -> str:
    """Une épaisseur de « Proposer » : la plus petite dalle qui tient pour ``h`` ; rend le tableau des épaisseurs déjà
    faites. Le résultat est gardé par le moteur jusqu'à ce que l'écran demande de le garder dans la note."""
    h = C.lire_nombre(h_txt)
    if h is None or round(h, 3) not in PR.epaisseurs(el):
        raise ValueError("épaisseur hors de la liste")
    h = round(h, 3)
    if not PR.jeux_complets(el):
        raise ValueError("aucun jeu de réactions")
    if not PR.k_scia(el):
        raise ValueError("aucun k coché « SCIA »")
    sig = _signature_proposition(el)
    lignes = _PROPOSITIONS.setdefault(sig, {})
    _PROPOSITIONS.move_to_end(sig)
    while len(_PROPOSITIONS) > 4:
        _PROPOSITIONS.popitem(last=False)
    if h not in lignes:
        lignes[h] = PP.pour_epaisseur(el, h)
    return SR.tableau_proposition(el, [lignes[x] for x in PR.epaisseurs(el) if x in lignes], complet=False)


# ------------------------------------------------------------------ le lien avec le module de raideur du sol (§ 9)
def _reprendre_sol(el: Dict, valeur) -> Optional[str]:
    """« Reprendre » : la note de raideur du sol choisie donne ses valeurs de k (la borne basse et la borne haute de sa
    fondation ouverte, cochées « SCIA ») et sa page de résultats, gardée dans le projet pour l'annexe de la note."""
    from . import sol as SOL
    if not isinstance(valeur, dict) or not isinstance(valeur.get("etat"), dict):
        raise ValueError("note de raideur du sol illisible")
    page = SOL.page_resultats(valeur["etat"])
    if not page.get("k"):
        raise ValueError(f"la note « {str(valeur.get('nom') or 'sans nom')[:60]} » ne donne pas de valeur de k")
    ks = list(dict.fromkeys(round(float(k), 2) for k in page["k"] if k and k > 0))
    el["sol"]["k"] = [{"k": k, "scia": True} for k in ks]
    el["sol"]["reprise"] = {"note": " ".join(str(valeur.get("nom") or "").split())[:120],
                            "chantier": " ".join(str(valeur.get("chantier") or "").split())[:120], "page": page}
    return None


# ------------------------------------------------------------------ le carnet : les grues enregistrées
def lire_carnet(carnet) -> List[Dict]:
    """Le carnet reçu de l'écran, relu : des grues {"nom", "grue": {chassis, a, b, appuis, rect, plots}, "reactions"} ;
    ce qui ne se relit pas est écarté (un châssis qui n'est plus dans la bibliothèque, une valeur hors bornes)."""
    out = []
    for x in carnet if isinstance(carnet, list) else []:
        try:
            nom = " ".join(str(x.get("nom") or "").split())[:60]
            el = PR.nouveau()
            el["grue"].update({k: copy.deepcopy(v) for k, v in (x.get("grue") or {}).items() if k in ("chassis", "a", "b", "appuis", "rect", "plots")})
            el["reactions"] = copy.deepcopy(x.get("reactions") or el["reactions"])
            relu = PR.lire(el)
        except (ValueError, AttributeError, TypeError, KeyError):
            continue
        if nom:
            out.append({"nom": nom, "grue": {k: relu["grue"][k] for k in ("chassis", "a", "b", "appuis", "rect", "plots")},
                        "reactions": relu["reactions"]})
    return out[:CARNET_MAX]


def _nom_de_grue(el: Dict) -> str:
    """Le nom sous lequel la grue du projet s'enregistre : le champ « Grue » du projet, sinon le châssis."""
    nom = " ".join((el["projet"].get("grue") or "").split())
    if nom:
        return nom[:60]
    ch = CH.lire(el["grue"])
    return (ch["nom"] or f"Châssis {C.fr(ch['a'], 2)} × {C.fr(ch['b'], 2)} m")[:60]


def _carnet(etat: Dict, el: Dict, ch: List[str], valeur) -> Optional[str]:
    carnet = lire_carnet(etat.get("carnet"))
    if ch == ["garder"]:
        nom = _nom_de_grue(el)
        grue = {"nom": nom, "grue": {k: copy.deepcopy(el["grue"][k]) for k in ("chassis", "a", "b", "appuis", "rect", "plots")},
                "reactions": copy.deepcopy(el["reactions"])}
        carnet = [x for x in carnet if x["nom"] != nom]
        if len(carnet) >= CARNET_MAX:
            raise ValueError(f"{CARNET_MAX} grues enregistrées au plus")
        etat["carnet"] = sorted(carnet + [grue], key=lambda x: x["nom"].lower())
        return f"Grue « {nom} » enregistrée"
    if ch == ["prendre"]:
        x = carnet[_indice(valeur, len(carnet))]
        el["grue"].update(copy.deepcopy(x["grue"]))
        el["reactions"] = copy.deepcopy(x["reactions"])
        if not el["projet"]["grue"]:
            el["projet"]["grue"] = x["nom"]
        return None
    if len(ch) == 2 and ch[1] == "-":
        i = _indice(ch[0], len(carnet))
        nom = carnet[i]["nom"]
        etat["carnet"] = carnet[:i] + carnet[i + 1:]
        return f"Grue « {nom} » retirée des grues enregistrées"
    raise ValueError("champ inconnu")


# ------------------------------------------------------------------ l'écran
def _dims(el: Dict) -> str:
    """« · 8,00 × 8,00 × 0,50 m » : ce qu'on lit d'une fondation dans la liste de la note (HTML)."""
    return f"· {SS.dimensions(el)}"


def resume_note(etat: Dict, synth: List[Dict]) -> Dict:
    """Le résumé de la NOTE : sa fondation la plus chargée face à la limite. Une fondation sans réaction (le poids propre
    seul) n'a pas de taux : la note est alors « Données incomplètes », jamais « Conforme » ; un calcul impossible passe
    devant tout."""
    def nom(i):
        return etat["fondations"][i].get("nom") or "Fondation"
    for niv in ("x", "n"):
        i = next((i for i, s in enumerate(synth) if s["niveau"] == niv), None)
        if i is not None:
            return {"taux": None, "niveau": niv, "statut": C.STATUTS[niv][1], "gouvernant": f"{nom(i)} · {synth[i]['phrase']}"}
    t, i = max(((s["taux"], -i) for i, s in enumerate(synth)))     # à taux égal, la première de la note
    i = -i
    niv = C.niveau(t)
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1], "gouvernant": f"{nom(i)} · {synth[i]['phrase']}"}


def arbre(etat: Dict, calc: Dict, s: Dict, carnet: List[Dict]) -> str:
    """La colonne de gauche COMPACTE (api_web/etabli_compact.arbre_compact, comme la poutre et le sol) : la ligne de la
    fondation ouverte (son nom, la liste des fondations de la note ; copier, supprimer), que l'écran pose dans l'en-tête
    du panneau, puis ses huit rubriques."""
    noms = [(f.get("nom") or f"F{i + 1}", _dims(f), False) for i, f in enumerate(etat["fondations"])]
    return E.arbre(GENRE, noms, etat["actif"], SS.formulaire(_f(etat), calc, s, carnet), "fondation", compact=True)


def resultat(etat: Dict) -> Dict:
    """Le calcul de la fondation ouverte et sa mise en forme : la colonne des vérifications, les fenêtres de calcul, le
    plan sous son en-tête, la colonne de gauche, le résumé de la note."""
    el = _f(etat)
    carnet = lire_carnet(etat.get("carnet"))
    calc = calculer(el)
    synth = [SR.synthese(f, calc if i == etat["actif"] else calculer(f)) for i, f in enumerate(etat["fondations"])]
    s = synth[etat["actif"]]
    html_res, volets = SR.colonne(el, calc, s)
    donnees = arbre(etat, calc, s, carnet)
    ef = efforts(el, calc, *SR.selection(el, calc)) if SR.carte_moments(el) and s.get("r_vue") is not None else None
    champ = SR.champ(el, calc, s, ef)
    vues = [{"cle": "plan", "titre": "Plan", "court": "Plan", "entete": SR.entete(el, calc, s), "echelle": False,
             "svg": DS.plan(el, calc["modele"], champ, legende=el["vue"]["carte"] != "aucune")}]
    nivs = [x["niveau"] for x in synth]
    statut = C.statut(next((n for n in ("x", "r", "n", "f") if n in nivs), "p")) if len(synth) > 1 else ""
    return {"resultat": html_res, "volets": volets, "vues": vues, "phrase": "", "statut_html": statut, "resume": resume_note(etat, synth),
            "donnees": donnees, "elements": ""}


# ------------------------------------------------------------------ l'export : les SAF (un par k coché) et la note
def controles_export(el: Dict, calc: Dict) -> Tuple[List[str], List[str]]:
    """(ce qui EMPÊCHE l'export, ce qui l'accompagne d'un avertissement) — cahier des charges § 8."""
    m = calc["modele"]
    ctl = m["controles"]
    bloque, avertit = [], []
    if not all(ctl["plaques_dedans"]):
        bloque.append("une surface d'appui sort de la dalle")
    if m["plots"] and not ctl["plots_ok"]:
        if not all(ctl["plots_dedans"]):
            bloque.append("un plot sort de la dalle")
        if not ctl["plots_separes"]:
            bloque.append("deux plots se chevauchent")
        if not all(ctl["plots_couvrent"]):
            bloque.append("une surface d'appui sort de son plot")
    if not m["cas"]:
        bloque.append("aucun jeu de réactions complet")
    if not PR.k_scia(el):
        bloque.append("aucun k coché « SCIA »")
    if all(ctl["plaques_dedans"]) and not ctl["distance_ok"]:
        avertit.append(f"une plaque est à {C.fr(ctl['distance'], 2)} m du bord, sous la distance mini de {C.fr(el['grue']['dmin'], 2)} m")
    return bloque, avertit


def _date(infos: Optional[Dict]) -> datetime.datetime:
    """La date de l'export : celle que la page donne (« JJ/MM/AAAA »), à défaut celle du jour."""
    try:
        j, mo, a = (int(x) for x in str((infos or {}).get("date") or "").split("/"))
        return datetime.datetime(a, mo, j)
    except (ValueError, TypeError):
        return datetime.datetime.combine(datetime.date.today(), datetime.time())


def projet_de_la_note(el: Dict, infos: Optional[Dict] = None) -> Dict:
    """Le projet tel que le SAF et la note l'écrivent (lot 4, § 2.1 : le bloc « Projet » a disparu) : le nom du projet et
    l'indice sont ceux de la barre du haut (le chantier et l'indice de la note), les dates celle de l'export ; une note
    d'avant le lot 4 garde ce qu'elle avait saisi quand la barre du haut ne dit rien. Le nom de la grue reste le sien
    (bloc 2)."""
    i = infos or {}
    p = dict(el["projet"])
    p["projet"] = (i.get("chantier") or "").strip() or p["projet"]
    p["indice"] = (i.get("indice") or "").strip() or p["indice"]
    jour = _date(infos).strftime("%d/%m/%Y")
    p["creation"] = p["creation"] or jour
    p["mise_a_jour"] = jour
    return dict(el, projet=p)


def fichiers_saf(el: Dict, calc: Dict, infos: Optional[Dict] = None) -> List[Tuple[str, bytes]]:
    """Le fichier SAF d'une fondation : [(nom, octets)] — UN SEUL (lot 4, § 7), écrit avec le PREMIER k coché « SCIA » ;
    le bureau fait varier k dans SCIA (la note rappelle les k à passer)."""
    nom_note = (infos or {}).get("nom") or el.get("nom") or TITRE
    date = _date(infos)
    k = EC.k_du_fichier(el)
    if k is None:
        return []
    ep = projet_de_la_note(el, infos)
    # (la feuille LLE_DONNEE porte le tableau de TOUS les k et le cas dimensionnant)
    return [(EC.nom_fichier(ep, nom_note), EC.fichier(ep, k, date, calc["par_k"], nom_note))]


def telecharger(etat: Dict, quoi: str, infos: Optional[Dict] = None) -> Optional[Dict]:
    """« export » : UN fichier .zip — le SAF de chaque fondation (un seul, le premier k coché « SCIA ») et la note PDF, le
    seul bouton d'export de l'écran ; « note_pdf » : la note seule (le bouton « Note PDF » de la page) ; « saf » : les
    SAF seuls, dans la même archive sans la note (les essais, le parcours du module — aucun bouton). Un export impossible
    est refusé avec sa raison (ValueError)."""
    if quoi == "note_pdf":
        from . import note_saf as NS
        return NS.fichier(etat, infos)
    if quoi not in ("export", "saf"):
        return None
    entrees: List[Tuple[str, bytes]] = []
    for el in etat["fondations"]:
        calc = calculer(el)
        bloque, _ = controles_export(el, calc)
        if bloque:
            raise ValueError(f"« {el.get('nom') or 'Fondation'} » ne s'exporte pas : {' ; '.join(bloque)}")
        prefixe = f"{C.nom_fichier(el.get('nom') or 'Fondation', '')}/" if len(etat["fondations"]) > 1 else ""
        entrees += [(prefixe + nom, octets) for nom, octets in fichiers_saf(el, calc, infos)]
    if quoi == "export":
        from . import note_saf as NS
        note = NS.fichier(etat, infos)
        entrees.append((note["nom"], base64.b64decode(note["base64"])))
    date = _date(infos)
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w", zipfile.ZIP_DEFLATED) as z:
        vus = set()
        for nom, octets in entrees:
            if nom in vus:                                        # deux fondations du même projet et du même k
                base, _, ext = nom.rpartition(".")
                nom = f"{base} ({sum(1 for v in vus if v.startswith(base))}).{ext}"
            vus.add(nom)
            z.writestr(zipfile.ZipInfo(nom, date_time=(date.year, date.month, date.day, 0, 0, 0)), octets)
    base = (infos or {}).get("nom") or (infos or {}).get("chantier") or _f(etat)["projet"]["projet"] or TITRE
    return {"nom": C.nom_fichier(f"SAF - {base}", "zip"), "type": TYPE_ZIP, "base64": base64.b64encode(tampon.getvalue()).decode("ascii")}


# ------------------------------------------------------------------ le parcours propre au module (api_web/parcours.py)
_JEUX = [{"action": "modifier", "chemin": ["reactions", "0", "P", str(i)], "valeur": v} for i, v in enumerate(("350", "440", "440", "350"))]


def _etat_type() -> Dict:
    """La dalle type du cahier des charges : trois jeux de réactions, tout le reste par défaut."""
    etat = nouvel_etat()
    etat["fondations"][0]["reactions"] = [{"nom": "", "P": list(p)} for p in ((350, 440, 440, 350), (150, 150, 550, 550), (135, 100, 800, 135))]
    return normaliser(etat)


def _avec(**k) -> Dict:
    """La dalle type, avec des données en plus ; un châssis de la bibliothèque y est CHOISI (ses appuis réels), comme à
    l'écran."""
    etat = _etat_type()
    el = etat["fondations"][0]
    for cle, v in k.items():
        if cle == "grue" and "chassis" in v:
            _poser_chassis(el, v["chassis"])
        el[cle].update(v)
    return normaliser(etat)


PARCOURS = _JEUX + [
    {"action": "calculer", "etat": _etat_type()},
    {"action": "selecteur", "role": "chassis"},
    {"action": "selecteur", "role": "chassis", "familles": ["Liebherr"]},
    {"action": "choisir", "role": "chassis", "nom": "Liebherr Châssis 132 HC – 4,6 m × 4,6 m"},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["grue", "position"], "valeur": [0.25, -0.5]},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["grue", "beta"], "valeur": 30},
    {"action": "modifier", "etat": _avec(grue={"repere": 1}), "chemin": ["grue", "cote", "perp"], "valeur": "2,5"},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["grue", "repere"], "valeur": "0"},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["dalle", "Lx"], "valeur": "10"},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["reactions", "+"], "valeur": "1"},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["reactions", "1", "-"], "valeur": "1"},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["sol", "k", "+"], "valeur": "1"},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["sol", "k", "4", "-"], "valeur": "1"},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["sol", "k", "3", "scia"], "valeur": True},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["vue", "carte"], "valeur": "w"},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["vue", "cas"], "valeur": "2.b"},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["vue", "k"], "valeur": "30"},
    {"action": "modifier", "etat": _avec(dalle={"forme": "contour", "sommets": [[0, 0], [12, 0], [12, 7], [7, 7], [7, 12], [0, 12]]}),
     "chemin": ["dalle", "sommet", "3"], "valeur": [8.0, 8.0]},
    {"action": "modifier", "etat": _avec(dalle={"forme": "contour", "sommets": [[0, 0], [12, 0], [12, 7], [7, 7], [7, 12], [0, 12]]}),
     "chemin": ["dalle", "sommet", "+"], "valeur": {"x": 6.0, "y": -1.0}},
    {"action": "modifier", "etat": _avec(dalle={"forme": "contour", "sommets": [[0, 0], [12, 0], [12, 7], [7, 7], [7, 12], [0, 12]]}),
     "chemin": ["dalle", "sommets", "2", "-"], "valeur": "1"},
    {"action": "modifier", "etat": _avec(grue={"plots": {"type": "coule", "a": 2.0, "b": 2.0, "h": 0.35}}), "chemin": ["grue", "plots", "a"], "valeur": "2,5"},
    {"action": "modifier", "etat": _avec(grue={"chassis": "liebherr-290hc-chassis-6000-pieds-a-vis"}), "chemin": ["options", "rond"], "valeur": "48"},
    {"action": "modifier", "etat": _avec(grue={"chassis": "wolff-krv-10-60-a60"}), "chemin": ["grue", "rect", "angle"], "valeur": "45"},
    {"action": "selecteur", "etat": _etat_type(), "role": "proposer:0.5"},
    {"action": "modifier", "etat": _etat_type(), "chemin": ["carnet", "garder"], "valeur": "1"},
    {"action": "telecharger", "etat": _etat_type(), "quoi": "saf"},
]

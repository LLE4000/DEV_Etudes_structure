# -*- coding: utf-8 -*-
"""Le module « Taux d'armature » du site, refait le 10/11/2026 sur la conception (docs/site/LOT_Q_QUATRE_MODULES.md § 3 ;
le moteur : ``beton/taux``, docs/beton/TAUX_ARMATURE_MOTEUR.md), sur l'ÉTABLI COMPACT — le modèle des cornières et de la
poutre (api_web/corniere.py, api_web/poutre.py, api_web/etabli_compact.py). Un QUANTITATIF : aucune vérification, la
note est en « Consultation ».

- à gauche, la ligne de la poutre ouverte (son nom, la liste des poutres de la note, copier, supprimer) puis ses quatre
  rubriques : 1 Poutre (la forme en pictogrammes, b × h, L et l'enrobage ; les talons de la forme qui en a),
  2 Armatures longitudinales (une carte compacte par ligne de barres, dépliable — le modèle des lits de la poutre :
  famille, lit, n, Ø, « de », « à » vides pour une barre filante, retours ; « + Lit inférieur », « + Lit supérieur »,
  « + Renfort », « + Barres de peau »), 3 Étriers (les TYPES, une carte par type avec ses lignes cadre / cadre
  intérieur / épingle et la case « cadre de talon » ; puis les ZONES, une carte par zone : de, à, pas, type, et son
  bilan en cours), 4 Hypothèses (secondaire : majoration, recouvrements, masses volumiques) ;
- au centre, les trois VUES de ``api_web/dessin_taux.py`` (élévation, coupe, bordereau) sous leur en-tête technique,
  qui rappelle la PROVENANCE d'une poutre reçue du module « Poutres béton armé » (« reçue de la poutre V1, le
  10/11/2026 », docs/site/LOT_Q_QUATRE_MODULES.md § 6.2) ; le CONTRAT DE SÉLECTION : chaque pièce du dessin
  (``data-sel`` : « beton », « b<i> », « t<i> », « z<i> ») a sa carte dans les données sous la même clé (``data-cible``) ;
- à droite, le VERDICT DE VALEUR (le taux en kg/m³, la masse d'acier majorée pour le volume de béton), trois blocs
  thématiques — « Acier par famille » (masse, part en jauge grise : ce n'est pas un taux de travail), « Par diamètre »
  (longueur, masse), « Béton » (section, volume, masse) — puis « Note entière » dès deux poutres ; les avertissements du
  moteur (le grave en tête) ; au pied, le ratio en kg par mètre de poutre, le bouton « Bordereau CSV » et la méthode ;
  quatre FENÊTRES DE CALCUL (repères et totaux, diamètres, béton, étriers par zone) dont chaque ligne « symbole =
  expression = valeurs = résultat » s'écrit à partir des seuls termes du moteur (tests/test_api_web_taux.py la
  réévalue).

Ce fichier MET EN FORME : il ne calcule rien (R3) ; il CHOISIT parmi ce que le moteur a nommé, COMPOSE les phrases et
FORMATE les nombres. Toute saisie passe par ``moteur.normaliser`` : hors domaine, elle est REFUSÉE avec le message du
moteur, et la poutre reste intacte ; les gestes (ajouter une ligne, une zone ; ce qui suit une limite de zones ou la
longueur) sont ceux de ``beton/taux/saisie.py``, et ce qui a SUIVI est dit au-dessus du résultat (jamais en silence).
Une note de l'ancien module (version 1) se rouvre : les noms des poutres sont gardés, chacune devient la poutre type,
et l'AVIS le dit une fois à l'ouverture.
"""
import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

from beton.taux import entree as EN
from beton.taux import moteur as MO
from beton.taux import saisie as SA

from . import commun as C
from . import dessin_taux as DT
from . import etabli_compact as E

MODULE = "taux"
PAGE = "taux-armature"
VERSION = 2                     # l'état de la note : {"version": 2, "poutres": [...], "actif": 0} (1 : l'ancien écran)
NOM_DEFAUT = "P1"
# deux clés de PASSAGE (posées par normaliser ou modifier, retirées et dites par resultat dans la même requête : elles
# ne sont jamais enregistrées) : l'avis d'ouverture d'une ancienne note, ce qui a suivi une saisie
CLE_AVIS = "avis_ouverture"
CLE_SUIVI = "suivi"
AVIS = {"v1": ("Cette note vient de l'ancien module « Taux d'armature » : ses poutres ont gardé leur nom et sont devenues la "
               "poutre type du nouveau module (le quantitatif est refait sur la norme : étriers, talons, recouvrements) — "
               "ressaisir leurs armatures.")}


def _note(**champs) -> Dict:
    """Une note neuve dont la poutre reçoit ``champs`` (pour le parcours)."""
    p = MO.nouvelle_poutre(NOM_DEFAUT)
    p.update(champs)
    return {"version": VERSION, "poutres": [p], "actif": 0}


_TALONS = {"talon_gauche": {"b": 15.0, "h": 20.0}, "talon_droit": {"b": 15.0, "h": 20.0}}
_RICHE = {"barres": [{"famille": "inf", "lit": 1, "n": 4, "d": 25, "de": None, "a": None, "retours": 2, "l_retour": None},
                     {"famille": "sup", "lit": 1, "n": 3, "d": 16, "de": None, "a": 2.0, "retours": 1, "l_retour": 40.0},
                     {"famille": "peau", "lit": None, "n": 2, "d": 10, "de": None, "a": None, "retours": 0, "l_retour": None}],
          "types_etriers": [{"nom": "E1", "lignes": [{"forme": "cadre", "d": 10, "largeur": None},
                                                     {"forme": "cadre_interieur", "d": 8, "largeur": 12.0},
                                                     {"forme": "epingle", "d": 8, "largeur": None}], "cadre_talon": True}],
          "zones": [{"de": 0.0, "a": 2.0, "s": 10.0, "type": "E1"}, {"de": 3.0, "a": 6.0, "s": 20.0, "type": "E1"}]}
# une note de l'ancien écran (version 1) : les noms gardés, les poutres neuves, l'avis
NOTE_V1 = {"version": 1, "poutres": [{"nom": "Poutre RDC", "forme": "talon_gauche", "L": 6.0, "b": 30.0, "h": 60.0,
                                      "familles": [{"role": "principal", "n": 3, "d": 20}]}], "actif": 0}
# la poutre « reçue » du module « Poutres béton armé » (son origine dite dans l'en-tête des vues)
_RECUE = {"origine": {"module": "poutre", "nom": "V1", "date": "10/11/2026"}}

# les requêtes de plus pour le parcours (api_web/parcours.py) et le relevé du paquet : les ajouts (lits, renfort, peau,
# type, ligne, zone), les formes à talons, les retours, les recouvrements (poutre de 14 m), une poutre riche (cadre
# intérieur, épingle, portion sans étriers), des refus (type employé retiré, zone de trop, recouvrement de zones), la
# note v1, la poutre reçue
PARCOURS = [
    {"action": "modifier", "chemin": ["barres", "+"], "valeur": "inf"},
    {"action": "modifier", "chemin": ["barres", "+"], "valeur": "sup"},
    {"action": "modifier", "chemin": ["barres", "+"], "valeur": "renfort"},
    {"action": "modifier", "chemin": ["barres", "+"], "valeur": "peau"},
    {"action": "modifier", "chemin": ["barres", 1, "-"], "valeur": "1"},
    {"action": "modifier", "chemin": ["barres", 0, "retours"], "valeur": "2"},
    {"action": "modifier", "chemin": ["types", "+"], "valeur": "1"},
    {"action": "modifier", "chemin": ["types", 0, "lignes", "+"], "valeur": "1"},
    {"action": "modifier", "chemin": ["types", 0, "-"], "valeur": "1"},
    {"action": "modifier", "chemin": ["zones", "+"], "valeur": "1"},
    {"action": "modifier", "chemin": ["zones", 1, "-"], "valeur": "1"},
    {"action": "modifier", "chemin": ["zones", 0, "a"], "valeur": "2"},
    {"action": "modifier", "chemin": ["zones", 0, "a"], "valeur": "5"},
    {"action": "modifier", "chemin": ["L"], "valeur": "14"},
    {"action": "modifier", "etat": _note(forme="deux_talons", **_TALONS), "chemin": ["types", 0, "cadre_talon"], "valeur": False},
    {"action": "modifier", "etat": _note(forme="talon_droit"), "chemin": ["talon_droit", "h"], "valeur": "25"},
    {"action": "modifier", "etat": _note(**_RICHE), "chemin": ["types", 0, "lignes", 1, "largeur"], "valeur": "14"},
    {"action": "modifier", "etat": _note(**_RICHE), "chemin": ["zones", "+"], "valeur": "1"},
    {"action": "modifier", "etat": _note(L=14.0), "chemin": ["recouvrements", "actifs"], "valeur": False},
    {"action": "modifier", "etat": _note(**_RECUE), "chemin": ["b"], "valeur": "35"},
    {"action": "calculer", "etat": NOTE_V1},
]


# ------------------------------------------------------------------ l'état de la note : des poutres
def nouvel_etat() -> Dict:
    """Une note neuve : la poutre TYPE de la conception (§ 3.5) — 30 × 60 cm, L = 6,00 m, c = 3 cm ; 3 Ø 20 inférieures
    filantes, un renfort 2 Ø 16 de 1,20 à 4,80 m, 2 Ø 12 supérieures filantes ; cadres Ø 8 au pas de 15 cm sur 1,50 m à
    chaque about et de 25 cm en travée : un vrai chiffre dès l'ouverture."""
    return {"version": VERSION, "poutres": [MO.nouvelle_poutre(NOM_DEFAUT)], "actif": 0}


def _indice(x) -> int:
    """L'indice de la poutre ouverte, lu sans faire tomber la note : un nombre entier fini, sinon la première."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return 0
    return int(x)


def _nom_v1(p, i: int) -> str:
    nom = p.get("nom") if isinstance(p, dict) else None
    nom = " ".join(str(nom).split()) if isinstance(nom, str) else ""
    return nom[:EN.NOM_MAX] or f"P{i + 1}"


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab, ou envoyé par « Poutres béton armé ») relu par le moteur ; une
    poutre illisible lève une erreur claire qui la nomme (le routeur la rend, l'écran l'affiche). Une note sans poutre
    est une note neuve ; une liste qui n'en est pas une est une note abîmée (refusée). Une note de l'ANCIEN module
    (``version`` autre que 2) : chaque poutre garde son nom et devient la poutre type, et l'avis d'ouverture est posé
    (``CLE_AVIS``, que ``resultat`` dit et retire). Une poutre reçue garde son ``origine``."""
    poutres = etat.get("poutres") if isinstance(etat, dict) else None
    if not poutres:
        return nouvel_etat()
    if not isinstance(poutres, list):
        raise ValueError("note abîmée : « poutres » doit être une liste")
    out = {"version": VERSION, "poutres": []}
    if etat.get("version") != VERSION:
        out["poutres"] = [MO.nouvelle_poutre(_nom_v1(p, i)) for i, p in enumerate(poutres)]
        out[CLE_AVIS] = "v1"
    else:
        for i, p in enumerate(poutres):
            if not isinstance(p, dict):
                raise ValueError(f"poutre n° {i + 1} illisible : un objet est attendu")
            try:
                out["poutres"].append(MO.normaliser(p))
            except ValueError as e:
                raise ValueError(f"poutre « {_nom_v1(p, i)} » : {e}") from None
    out["actif"] = min(max(0, _indice(etat.get("actif"))), len(out["poutres"]) - 1)
    return out


def _poutre(etat: Dict) -> Dict:
    return etat["poutres"][etat["actif"]]


def _nom_libre(ps: List[Dict]) -> str:
    pris = {p.get("nom") for p in ps}
    k = len(ps) + 1
    while f"P{k}" in pris:
        k += 1
    return f"P{k}"


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les poutres de la note : en ouvrir une, en ajouter une (la poutre type), copier l'ouverte (ou ``index`` : la copie
    s'insère juste après), en supprimer une (jamais la dernière)."""
    ps = etat["poutres"]
    if action == "activer":
        etat["actif"] = min(max(0, _indice(index)), len(ps) - 1)
    elif action == "ajouter":
        ps.append(MO.nouvelle_poutre(_nom_libre(ps)))
        etat["actif"] = len(ps) - 1
    elif action == "dupliquer":
        i = etat["actif"] if index is None else min(max(0, _indice(index)), len(ps) - 1)
        p = copy.deepcopy(ps[i])
        p["nom"] = f"{p.get('nom') or 'Poutre'} (copie)"[:EN.NOM_MAX]
        ps.insert(i + 1, p)
        etat["actif"] = i + 1
    elif action == "supprimer" and len(ps) > 1:
        i = min(max(0, _indice(index)), len(ps) - 1)
        del ps[i]
        if etat["actif"] > i or etat["actif"] >= len(ps):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
NUMERIQUES = ("b", "h", "L", "enrobage", "majoration", "rho_beton", "rho_acier")
TALONS = ("talon_gauche", "talon_droit")


def _nombre(valeur) -> float:
    x = C.lire_nombre(valeur)
    if x is None:
        raise ValueError(f"nombre illisible : « {str(valeur)[:40]} »")
    return x


def _entier(valeur):
    """Un entier saisi (n, lit, Ø, retours) : 3 ; « 3 » ; 3,0 — un nombre non entier est passé tel quel, le moteur le
    refuse en nommant le champ."""
    x = _nombre(valeur)
    return int(x) if float(x).is_integer() else x


def _case(valeur) -> bool:
    v = {True: True, False: False, "true": True, "false": False, "True": True, "False": False, 1: True, 0: False}.get(valeur)
    if v is None:
        raise ValueError(f"case : « {str(valeur)[:20]} » (coché ou non)")
    return v


def _rang(liste: list, x, quoi: str) -> int:
    try:
        i = int(x)
    except (TypeError, ValueError):
        raise ValueError(f"{quoi} : rang « {str(x)[:12]} » illisible") from None
    if not 0 <= i < len(liste):
        raise ValueError(f"{quoi} n° {i + 1} : il n'y en a que {len(liste)}")
    return i


def _appliquer(p: Dict, ch: List, valeur) -> List[str]:
    """Écrit la saisie ``valeur`` au chemin ``ch`` de la poutre ``p`` (une copie) ; rend ce qui a SUIVI, en mots. Lève
    ValueError (le message du moteur, ou le sien) pour une saisie illisible ; le moteur relit ensuite la poutre."""
    k = ch[0]
    if len(ch) == 1:
        if k == "nom":
            nom = str(valeur if valeur is not None else "").strip()
            if len(nom) > EN.NOM_MAX:
                raise ValueError(f"nom trop long : {len(nom)} caractères (au plus {EN.NOM_MAX})")
            p["nom"] = nom
        elif k == "forme":
            p["forme"] = valeur
        elif k == "L":
            return SA.changer_longueur(p, _nombre(valeur))
        elif k in NUMERIQUES:
            p[k] = _nombre(valeur)
        else:
            raise KeyError(k)
        return []
    if k in TALONS and len(ch) == 2 and ch[1] in ("b", "h"):
        p[k][ch[1]] = _nombre(valeur)
        return []
    if k == "recouvrements" and len(ch) == 2:
        if ch[1] == "actifs":
            p["recouvrements"]["actifs"] = _case(valeur)
        elif ch[1] in ("l_com", "k"):
            p["recouvrements"][ch[1]] = _nombre(valeur)
        else:
            raise KeyError(ch[1])
        return []
    if k == "barres":
        if ch[1] == "+":
            SA.ajouter_barre(p, valeur)
            return []
        i = _rang(p["barres"], ch[1], "barres")
        b = p["barres"][i]
        champ = ch[2] if len(ch) == 3 else None
        if champ == "-":
            del p["barres"][i]
        elif champ == "famille":
            b["famille"] = valeur
        elif champ in ("lit", "n", "d", "retours"):
            b[champ] = _entier(valeur)
        elif champ in ("de", "a", "l_retour"):
            b[champ] = C.lire_auto(valeur)
        else:
            raise KeyError(champ)
        return []
    if k == "types":
        if ch[1] == "+":
            SA.ajouter_type(p)
            return []
        i = _rang(p["types_etriers"], ch[1], "types d'étriers")
        t = p["types_etriers"][i]
        if len(ch) == 3 and ch[2] == "-":
            SA.retirer_type(p, i)
        elif len(ch) == 3 and ch[2] == "nom":
            SA.renommer_type(p, i, "" if valeur is None else valeur)
        elif len(ch) == 3 and ch[2] == "cadre_talon":
            t["cadre_talon"] = _case(valeur)
        elif len(ch) == 4 and ch[2] == "lignes" and ch[3] == "+":
            SA.ajouter_ligne(p, i)
        elif len(ch) == 5 and ch[2] == "lignes":
            j = _rang(t["lignes"], ch[3], f"type « {t['nom']} » · lignes")
            champ = ch[4]
            if champ == "-":
                SA.retirer_ligne(p, i, j)
            elif champ == "forme":
                SA.changer_forme_ligne(p, i, j, valeur)
            elif champ == "d":
                t["lignes"][j]["d"] = _entier(valeur)
            elif champ == "largeur":
                t["lignes"][j]["largeur"] = _nombre(valeur)
            else:
                raise KeyError(champ)
        else:
            raise KeyError("/".join(str(x) for x in ch))
        return []
    if k == "zones":
        if ch[1] == "+":
            SA.ajouter_zone(p)
            return []
        i = _rang(p["zones"], ch[1], "zones")
        champ = ch[2] if len(ch) == 3 else None
        if champ == "-":
            del p["zones"][i]
        elif champ in ("de", "a"):
            return SA.deplacer_borne(p, i, champ, _nombre(valeur))
        elif champ == "s":
            p["zones"][i]["s"] = _nombre(valeur)
        elif champ == "type":
            p["zones"][i]["type"] = valeur
        else:
            raise KeyError(champ)
        return []
    raise KeyError(k)


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la poutre ouverte. Le moteur relit la poutre : une valeur hors de son domaine est REFUSÉE
    avec son message (jamais ramenée en silence), et la poutre reste intacte. Ce qui a SUIVI la saisie (la zone jointive
    d'une limite déplacée, les abscisses de la moitié droite quand la longueur change) est posé pour le résultat
    (``CLE_SUIVI``), qui le dit."""
    ch = list(chemin) if isinstance(chemin, (list, tuple)) else []
    if not ch:
        return "Champ inconnu : « » (chemin vide)"
    p = copy.deepcopy(_poutre(etat))
    try:
        suivis = _appliquer(p, ch, valeur)
        etat["poutres"][etat["actif"]] = MO.normaliser(p)
    except KeyError:
        return f"Champ inconnu : « {'/'.join(str(x) for x in ch)} »"
    except (ValueError, TypeError, IndexError) as e:
        return f"Saisie refusée : {e}"
    if suivis:
        etat[CLE_SUIVI] = suivis
    return None


# ------------------------------------------------------------------ les nombres
DECIMALES = {"m": 3, "kg": 2, "kg/m": 3, "kg/m³": 1, "m³": 3, "cm²": 1, "cm": 1, "mm": 0, "m²": 4, "%": 1}


def _res(x: float, unite: str) -> str:
    """Un RÉSULTAT de formule et son unité : longueurs au millimètre, masses au centième de kg, masses linéiques au
    gramme par mètre, le taux au dixième ; un nombre sans unité entier s'écrit entier (un nombre de cours), sinon au
    millième (un quotient)."""
    if not unite:
        return C.fr(x, 0) if float(x).is_integer() else C.fr(x, 3)
    return C.qte(x, unite, DECIMALES.get(unite, 2))


def _v(x: float) -> str:
    """Une VALEUR substituée dans une formule, sans la dénaturer : un entier reste entier, une longueur garde le
    millimètre (quatre chiffres après la virgule sous 1), une grande valeur le dixième."""
    if float(x).is_integer():
        return C.nb(x, 0)
    a = abs(x)
    return C.nb(round(x, 1), 1) if a >= 1000 else (C.nb(round(x, 3), 3) if a >= 1 else C.nb(round(x, 4), 4))


def _m(x: float) -> str:
    """Une abscisse ou une longueur de la poutre : « 1,50 m » (deux décimales, trois s'il le faut)."""
    nd = 2 if abs(x * 100 - round(x * 100)) < 1e-6 else 3
    return C.qte(x, "m", nd)


def _x(x: float) -> str:
    """Une abscisse sans son unité, au format de ``_m`` : « 1,50 »."""
    return C.fr(x, 2 if abs(x * 100 - round(x * 100)) < 1e-6 else 3)


def _part(part: float) -> int:
    """La part d'une famille dans la masse d'acier, en % entier (une JAUGE grise, pas un taux de travail : elle n'est
    pas ramenée dans une bande de couleur comme ``commun.pct``)."""
    return int(C.fr(part * 100, 0).replace(C.FINE, ""))


def _cm(x: float) -> str:
    return C.nb(x, 1)


# ------------------------------------------------------------------ les mots
FACES = {"inf": "inférieur", "sup": "supérieur", "peau": "de peau"}
SORTES_AJOUT = (("inf", "Lit inférieur"), ("sup", "Lit supérieur"), ("renfort", "Renfort"), ("peau", "Barres de peau"))
RETOURS_TXT = ((0, "aucun"), (1, "un bout"), (2, "deux bouts"))
# la forme de la section : sa clé, le mot court du segment, le pictogramme (l'âme et ses talons bas, dans 26 × 18)
_PICTOS_FORME = {"rectangulaire": "M9 2H17V16H9Z", "talon_gauche": "M10 2H17V16H3V11H10Z",
                 "talon_droit": "M9 2H16V11H23V16H9Z", "deux_talons": "M10 2H16V11H23V16H3V11H10Z"}


def _picto_forme(forme: str) -> str:
    return (f'<svg class="tx-pic" viewBox="0 0 26 18" aria-hidden="true"><path d="{_PICTOS_FORME[forme]}" fill="currentColor" '
            f'fill-opacity=".18" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/></svg>')


def _segments_forme(p: Dict) -> str:
    """La forme en quatre segments à pictogramme ; le libellé entier au survol (commun.segments n'a pas d'aide par
    bouton, ni de pictogramme)."""
    bt = "".join(f'<button type="button" class="{"on" if f == p["forme"] else ""}" data-chemin="forme" data-sorte="choix" '
                 f'data-valeur="{f}" title="{C.esc(EN.LIBELLES_FORME[f].capitalize())}">{_picto_forme(f)}</button>'
                 for f in EN.FORMES)
    return f'<div class="seg full tx-formes">{bt}</div>'


def _dims(p: Dict) -> str:
    """« 30 × 60 cm · L 6,00 m » (HTML : les quantités ne se coupent pas)."""
    return f"{C.nb(p['b'])}{C.NBSP}×{C.NBSP}{C.qte(p['h'], 'cm', 0) if float(p['h']).is_integer() else C.nb(p['h']) + C.NBSP + 'cm'} · L{C.NBSP}{_m(p['L'])}"


def _origine_txt(p: Dict) -> str:
    """« reçue de la poutre V1, le 10/11/2026 » (texte ; sans date quand l'envoi n'en porte pas) ; « » sinon."""
    o = p.get("origine")
    if not o:
        return ""
    return f"reçue de la poutre {o['nom'] or '(sans nom)'}" + (f", le {o['date']}" if o.get("date") else "")


# ------------------------------------------------------------------ la colonne de gauche : les rubriques
def _rubrique_poutre(p: Dict) -> str:
    s = C.ligne("Forme", _segments_forme(p), "", "La forme de la section : rectangulaire, un talon à gauche, un talon à droite ou "
                "deux talons — les talons sont BAS, à l'extérieur de l'âme ; leur béton est compté, leur cadre aussi (case « cadre "
                "de talon » des types d'étriers).", full=True)
    s += f'<div class="hint tx-forme-n">{C.esc(EN.LIBELLES_FORME[p["forme"]].capitalize())}</div>'
    s += C.ligne2(("b largeur", C.num(["b"], p["b"], "cm"), "La largeur de l'âme."),
                  ("h hauteur", C.num(["h"], p["h"], "cm"), "La hauteur totale de la poutre."))
    s += C.ligne2(("L longueur", C.num(["L"], p["L"], "m"), "La longueur de la poutre, d'about à about. Ce qui se compte depuis l'about "
                   "droit (les abscisses au-delà de L/2 : la fin d'un renfort, la limite d'une zone) garde sa distance à l'about droit."),
                  ("c enrobage", C.num(["enrobage"], p["enrobage"], "cm"), "L'enrobage des aciers, aux faces ET aux abouts : une barre filante "
                   "mesure L − 2c, un cadre a le contour hors tout à l'enrobage."))
    for cote in EN.talons(p["forme"]):
        t = p[cote]
        nom = EN.LIBELLES_TALON[cote]
        ind = "g" if cote == "talon_gauche" else "d"
        s += C.ligne2((f"b<sub>t,{ind}</sub> {C.esc(nom)}", C.num([cote, "b"], t["b"], "cm"), f"La largeur du {nom}, hors de l'âme."),
                      (f"h<sub>t,{ind}</sub>", C.num([cote, "h"], t["h"], "cm"), f"La hauteur du {nom}, depuis le dessous de la poutre."))
    return f'<div class="et-piece et-plat" data-cible="beton">{s}</div>'


def _carte_barre(i: int, b: Dict, rp: Dict) -> str:
    """Une ligne de barres : la ligne courte (son repère, n, Ø, « filant · inf. L1 »), puis son détail (famille, lit,
    de, à, retours) — dépliée quand elle est choisie, ici ou sur le dessin."""
    ch = ["barres", i]
    peau = b["famille"] == "peau"
    # court (la carte est rangée sous sa face) : « filant · L1 », « 1,20 → 4,80 m · L2 », « filant · par face »
    if b["de"] is None and b["a"] is None:
        pos = "filant"
    else:
        pos = f"{'about' if b['de'] is None else _x(b['de'])} → {'about' if b['a'] is None else _m(b['a'])}"
    txt = f"{pos} · {'par face' if peau else 'L' + str(b['lit'])}" if pos == "filant" else pos
    tete = (f'<div class="et-l1"><button type="button" class="et-tag" data-deplier="1" title="Repère {rp["repere"]} : détail">{rp["repere"]}</button>'
            f'{C.num(ch + ["n"], b["n"], "", entier=True, mini=1)}'
            f'{C.choix(ch + ["d"], b["d"], [(x, f"Ø{C.NBSP}{x}") for x in EN.DIAMETRES])}'
            f'<span class="et-txt" title="{C.esc(rp["designation"] + " · " + txt)}">{C.esc(txt)}</span>'
            f'<button type="button" class="et-dep" data-deplier="1" title="Famille, étendue, retours">{C.icon("chevron-down")}</button></div>')
    det = C.ligne("Famille", C.segments(ch + ["famille"], b["famille"], [(f, FACES[f].capitalize() if f != "peau" else "Peau") for f in EN.FAMILLES]),
                  full=True)
    if not peau:
        det += C.ligne("Lit", C.num(ch + ["lit"], b["lit"], "", entier=True, mini=1), "",
                       "Le rang du lit sur sa face : 1 contre le cadre, 2 au-dessus (au-dessous pour la face supérieure)… Il place la barre "
                       "sur la coupe et l'élévation ; il ne change aucune longueur.")
    det += C.ligne2(("de", C.num(ch + ["de"], b["de"], "m", auto="about"), "L'abscisse du bout gauche, depuis l'about gauche ; vide : "
                     "l'about (à l'enrobage). « de » et « à » vides : une barre FILANTE, de c à L − c."),
                    ("à", C.num(ch + ["a"], b["a"], "m", auto="about"), "L'abscisse du bout droit ; vide : l'about droit (à l'enrobage)."))
    det += C.ligne("Retours", C.segments(ch + ["retours"], b["retours"], RETOURS_TXT), "",
                   "Le nombre de bouts relevés (retours d'ancrage), comptés dans la longueur unitaire.", full=True)
    if b["retours"]:
        auto = f"auto · {_cm(rp['detail']['l_retour'])}" if rp["detail"]["l_retour_auto"] else "auto"
        det += C.ligne("l<sub>r</sub> retour", C.num(ch + ["l_retour"], b["l_retour"], "cm", auto=auto), "",
                       "La longueur d'un retour ; vide : toute la hauteur entre enrobages (h − 2c ; b − 2c pour la peau) — le côté qui "
                       "commande le plus d'acier.")
    det += f'<div class="hint et-info">{C.esc(rp["designation"])} · {C.qte(rp["l_unitaire"], "m", 3)} par barre · {C.qte(rp["masse"], "kg", 2)}</div>'
    det += E.retirer(ch + ["-"], "Retirer cette ligne")
    return f'<div class="et-piece tx-carte" data-cible="b{i}">{tete}<div class="et-det">{det}</div></div>'


GROUPES_BARRES = (("inf", "Face inférieure"), ("sup", "Face supérieure"), ("peau", "Barres de peau"))


def _rubrique_longitudinales(p: Dict, res: Dict) -> str:
    reps = {r["cle"]: r for r in res["reperes"]}
    out = ""
    for fam, titre in GROUPES_BARRES:
        cartes = "".join(_carte_barre(i, b, reps[f"b{i}"]) for i, b in enumerate(p["barres"]) if b["famille"] == fam)
        if cartes:
            masse = sum(reps[f"b{i}"]["masse"] for i, b in enumerate(p["barres"]) if b["famille"] == fam)
            out += (f'<div class="et-grp"><div class="et-gh"><span class="t">{titre}</span><span class="et-req">'
                    f'<b>{C.fr(masse, 1)}</b>{C.NBSP}kg</span></div>{cartes}</div>')
    if not out:
        out = '<div class="hint">Aucune barre longitudinale : le bordereau ne compte que les étriers.</div>'
    ajouts = "".join(E.plus(["barres", "+"], k, t) for k, t in SORTES_AJOUT) if len(p["barres"]) < EN.MAX_BARRES else ""
    return out + f'<div class="et-pied tx-ajouts">{ajouts}</div>'


def _carte_type(i: int, t: Dict, rt: Dict, forme: str) -> str:
    ch = ["types", i]
    n = f"{rt['n']}{C.NBSP}cours" if rt["employe"] else "inemployé"
    tete = (f'<div class="et-l1"><button type="button" class="et-tag c" data-deplier="1" title="Le type {C.esc(t["nom"])} : ses lignes">'
            f'{C.esc(t["nom"])}</button><span class="et-txt" title="{C.esc(rt["resume"])}">{C.esc(rt["resume"])}</span>'
            f'<span class="tx-bilan{"" if rt["employe"] else " ko"}">{n}</span>'
            f'<button type="button" class="et-dep" data-deplier="1" title="Lignes du type">{C.icon("chevron-down")}</button></div>')
    det = C.ligne("Nom", C.texte(ch + ["nom"], t["nom"], "E1"), "", "Le nom du type ; les zones le désignent par lui (elles suivent un "
                  "renommage).")
    for j, (li, rl) in enumerate(zip(t["lignes"], rt["lignes"])):
        cl = ch + ["lignes", j]
        corps = C.ligne2(("Forme", C.choix(cl + ["forme"], li["forme"], [(f, EN.LIBELLES_LIGNE[f]) for f in EN.FORMES_LIGNE]),
                          "Cadre : le contour de l'âme à l'enrobage. Cadre intérieur : un cadre plus étroit, de largeur hors tout w. Épingle : "
                          "une branche verticale à deux retours."),
                         ("Ø", C.choix(cl + ["d"], li["d"], [(x, f"Ø{C.NBSP}{x}") for x in EN.DIAMETRES_ETRIERS]), ""))
        if li["forme"] == "cadre_interieur":
            corps += C.ligne("w largeur", C.num(cl + ["largeur"], li["largeur"], "cm"), "",
                             "La largeur hors tout du cadre intérieur ; au plus celle du cadre de l'âme (b − 2c).")
        retirer = E.retirer(cl + ["-"], "Retirer la ligne") if len(t["lignes"]) > 1 else ""
        det += (f'<div class="tx-ligne"><div class="tx-ligne-t">Ligne {j + 1} · {C.esc(EN.LIBELLES_LIGNE[li["forme"]])} · '
                f'{C.qte(rl["l_unitaire"], "m", 3)}</div>{corps}{retirer}</div>')
    if len(t["lignes"]) < EN.MAX_LIGNES:
        det += f'<div class="et-pied">{E.plus(ch + ["lignes", "+"], "1", "Ligne")}</div>'
    if EN.a_talon(forme):
        det += C.case(ch + ["cadre_talon"], t["cadre_talon"], "cadre de talon à chaque cours")
    det += E.retirer(ch + ["-"], "Retirer ce type")
    return f'<div class="et-piece tx-carte" data-cible="t{i}">{tete}<div class="et-det">{det}</div></div>'


def _carte_zone(i: int, z: Dict, rz: Dict, noms: List[str]) -> str:
    ch = ["zones", i]
    ferm = " + 1 de fermeture" if rz["fermeture"] else ""
    txt = f"{_x(z['de'])} → {_m(z['a'])} · e{C.NBSP}{_cm(z['s'])} · {z['type']}"
    tete = (f'<div class="et-l1"><button type="button" class="et-tag c" data-deplier="1" title="Zone {rz["nom"]} : détail">{rz["nom"]}</button>'
            f'<span class="et-txt">{C.esc(txt)}</span><span class="tx-bilan">{rz["n"]}{C.NBSP}étrier{"s" if rz["n"] > 1 else ""}</span>'
            f'<button type="button" class="et-dep" data-deplier="1" title="De, à, pas, type">{C.icon("chevron-down")}</button></div>')
    det = C.ligne2(("de", C.num(ch + ["de"], z["de"], "m"), "Le début de la zone, depuis l'about gauche. La zone jointive qui finissait là "
                    "suit : c'est la limite des deux zones qui se déplace."),
                   ("à", C.num(ch + ["a"], z["a"], "m"), "La fin de la zone. La zone jointive qui commençait là suit."))
    det += C.ligne2(("e pas", C.num(ch + ["s"], z["s"], "cm"), "L'espacement des cours, le long de la poutre."),
                    ("Type", C.choix(ch + ["type"], z["type"], [(n, n) for n in noms]), "Ce qu'on pose à chaque cours (rubrique Types)."))
    det += (f'<div class="hint et-info">l<sub>z</sub>/e = {C.esc(_v(rz["longueur"]))}/{C.esc(_v(z["s"] / 100.0))} = {C.esc(_v(rz["quotient"]))} → '
            f'{rz["cours"]}{C.NBSP}cours{ferm}</div>')
    det += E.retirer(ch + ["-"], "Retirer cette zone")
    return f'<div class="et-piece tx-carte" data-cible="z{i}">{tete}<div class="et-det">{det}</div></div>'


def _rubrique_etriers(p: Dict, res: Dict) -> str:
    noms = [t["nom"] for t in p["types_etriers"]]
    types = "".join(_carte_type(i, t, res["types"][i], p["forme"]) for i, t in enumerate(p["types_etriers"]))
    zones = "".join(_carte_zone(i, z, res["zones"][i], noms) for i, z in enumerate(p["zones"]))
    n = sum(z["n"] for z in res["zones"])
    out = (f'<div class="et-grp"><div class="et-gh"><span class="t">Types</span><span class="et-req">ce qu\'on pose à chaque cours</span></div>'
           f'{types or "<div class=hint>Aucun type : « + Type » en crée un (un cadre).</div>"}'
           f'<div class="et-pied">{E.plus(["types", "+"], "1", "Type") if len(noms) < EN.MAX_TYPES else ""}</div></div>')
    trous = "".join(C.message(f"Sans étriers de {C.esc(_m(x1))} à {C.esc(_m(x2))}", True) for x1, x2 in res["trous"])
    out += (f'<div class="et-grp"><div class="et-gh"><span class="t">Zones</span><span class="et-req"><b>{n}</b>{C.NBSP}étriers</span></div>'
            f'{zones or "<div class=hint>Aucune zone : « + Zone » en propose une.</div>"}{trous}'
            f'<div class="et-pied">{E.plus(["zones", "+"], "1", "Zone") if res["trous"] and len(p["zones"]) < EN.MAX_ZONES else ""}'
            f'<span class="et-pied-a">de, à en m · pas en cm</span></div></div>')
    return out


def _rubrique_hypotheses(p: Dict) -> str:
    r = p["recouvrements"]
    s = C.ligne("Majoration", C.num(["majoration"], p["majoration"], "%"), "",
                "Chutes et ligatures : la masse d'acier est majorée de ce pourcentage ; le taux et le ratio au mètre en tiennent compte.")
    s += C.case(["recouvrements", "actifs"], r["actifs"], "compter les recouvrements des barres trop longues")
    s += C.ligne2(("l<sub>com</sub>", C.num(["recouvrements", "l_com"], r["l_com"], "m"), "La longueur commerciale des barres : une barre "
                   "plus longue est faite de plusieurs, qui se recouvrent."),
                  ("k × Ø", C.num(["recouvrements", "k"], r["k"], "Ø"), "La longueur d'un recouvrement, en diamètres."))
    s += C.ligne2(("ρ<sub>c</sub> béton", C.num(["rho_beton"], p["rho_beton"], "kg/m³"), "La masse volumique du béton armé (la masse du béton)."),
                  ("ρ<sub>s</sub> acier", C.num(["rho_acier"], p["rho_acier"], "kg/m³"), "La masse volumique de l'acier : la masse linéique "
                   "d'une barre en vient (π·Ø²/4·ρ_s)."))
    return s


def _resume_longitudinales(p: Dict) -> str:
    if not p["barres"]:
        return "aucune barre"
    lib = [EN.libelle_barres(b["n"], b["d"]) for b in p["barres"][:3]]
    return C.esc(", ".join(lib) + (" …" if len(p["barres"]) > 3 else ""))


def formulaire(etat: Dict, res: Optional[Dict] = None) -> str:
    """Les quatre rubriques de la poutre ouverte : 1 Poutre, 2 Armatures longitudinales, 3 Étriers, 4 Hypothèses
    (secondaire). ``res`` : le calcul de la poutre ouverte (``MO.calculer``), refait s'il manque."""
    p = _poutre(etat)
    res = res or MO.calculer(p)
    n = sum(z["n"] for z in res["zones"])
    r1 = f"{C.esc(EN.LIBELLES_FORME[p['forme']])} · {_dims(p)} · c{C.NBSP}{C.nb(p['enrobage'])}{C.NBSP}cm"
    r3 = f"{len(p['types_etriers'])} type{'s' if len(p['types_etriers']) > 1 else ''} · {len(p['zones'])} zone{'s' if len(p['zones']) > 1 else ''} · {n}{C.NBSP}étriers"
    rec = p["recouvrements"]
    r4 = (f"+{C.nb(p['majoration'])}{C.NBSP}% · " + (f"recouvrements {C.nb(rec['k'])}{C.NBSP}Ø au-delà de {C.qte(rec['l_com'], 'm', 0)}"
                                                    if rec["actifs"] else "sans recouvrements"))
    return (C.section(1, "Poutre", r1, _rubrique_poutre(p), True, "pt")
            + C.section(2, "Armatures longitudinales", _resume_longitudinales(p), _rubrique_longitudinales(p, res), True, "lo")
            + C.section(3, "Étriers", r3, _rubrique_etriers(p, res), True, "et")
            + C.section(4, "Hypothèses", r4, _rubrique_hypotheses(p), False, "hy", secondaire=True))


def arbre(etat: Dict, res: Optional[Dict] = None) -> str:
    """La colonne de gauche COMPACTE (etabli_compact.arbre_compact) : la ligne de la poutre ouverte, puis ses rubriques."""
    elements = [(p.get("nom") or f"P{i + 1}", f"· {C.nb(p['b'])} × {C.nb(p['h'])} · {_m(p['L'])}", False) for i, p in enumerate(etat["poutres"])]
    return E.arbre(E.POUTRE, elements, etat["actif"], formulaire(etat, res), "poutre", compact=True)


# ------------------------------------------------------------------ les formules : « symbole = expression = valeurs = résultat »
def _fx(f: Dict) -> Dict:
    """Une formule du moteur, écrite à partir de SES SEULS TERMES (dans leur ordre) ; une conversion d'unités est faite
    dans les valeurs (cm et mm en m), et l'unité de chaque valeur substituée est écrite quand elle n'est pas celle du
    résultat. tests/test_api_web_taux.py relit les valeurs écrites et refait le calcul."""
    t = {x["symbole"]: x["valeur"] for x in f["termes"]}
    k, s = f["cle"], f["symbole"]

    def v(nom, div=1.0):
        return _v(t[nom] / div)
    if k == "x1":
        parts = [s, "max(de ; c)", f"max({v('de')} ; {v('c')})"]
    elif k == "x2":
        parts = [s, "min(à ; L − c)", f"min({v('à')} ; {v('L')} − {v('c')})"]
    elif k == "l_droite":
        parts = [s, "x_2 − x_1", f"{v('x_2')} − {v('x_1')}"]
    elif k == "l_retour_auto":
        cote = "h" if "h" in t else "b"
        parts = [s, f"{cote} − 2·c", f"{v(cote)} − 2 × {v('c')}"]
    elif k == "l_dev":
        parts = [s, "l_d + r·l_r", f"{v('l_d')} + {v('r')} × {v('l_r', 100.0)}"]
    elif k == "l_rec":
        parts = [s, "k·Ø", f"{v('k')} × {v('Ø', 1000.0)}"]
    elif k == "n_rec":
        parts = [s, "⌈(l_0 − l_com) / (l_com − l_rec)⌉", f"⌈({v('l_0')} − {v('l_com')}) / ({v('l_com')} − {v('l_rec')})⌉"]
    elif k == "l_unitaire":
        e, w = ["l_d"], [v("l_d")]
        if "r" in t:
            e.append("r·l_r")
            w.append(f"{v('r')} × {v('l_r', 100.0)}")
        if "n_rec" in t:
            e.append("n_rec·l_rec")
            w.append(f"{v('n_rec')} × {v('l_rec')}")
        parts = [s, " + ".join(e), " + ".join(w)] if len(e) > 1 else [s, e[0], w[0]]
    elif k == "n_peau":
        parts = [s, "n par face × faces", f"{v('n_face')} × {v('faces')}"]
    elif k == "l_totale":
        parts = [s, "n·l", f"{v('n')} × {v('l')}"]
    elif k == "m_lin":
        parts = [s, "π·Ø²/4·ρ_s", f"π × {v('Ø', 1000.0)}² / 4 × {v('ρ_s')}"]
    elif k == "masse":
        parts = [s, "L·m", f"{v('L')} × {v('m')}"]
    elif k == "l_r_etrier":
        parts = [s, f"max({C.nb(MO.RETOUR_K)}·Ø ; {C.nb(MO.RETOUR_MIN)} mm)", f"max({C.nb(MO.RETOUR_K)} × {v('Ø')} ; {C.nb(MO.RETOUR_MIN)})"]
    elif k == "contour_cadre":
        parts = [s, "2·[(b − 2c) + (h − 2c)]", f"2 × [({v('b', 100.0)} − 2 × {v('c', 100.0)}) + ({v('h', 100.0)} − 2 × {v('c', 100.0)})]"]
    elif k == "contour_interieur":
        parts = [s, "2·[w + (h − 2c)]", f"2 × [{v('w', 100.0)} + ({v('h', 100.0)} − 2 × {v('c', 100.0)})]"]
    elif k == "contour_epingle":
        parts = [s, "h − 2c", f"{v('h', 100.0)} − 2 × {v('c', 100.0)}"]
    elif k == "contour_talon":
        parts = [s, "2·[(b_tot − 2c) + (h_t − 2c)]", f"2 × [({v('b_tot', 100.0)} − 2 × {v('c', 100.0)}) + ({v('h_t', 100.0)} − 2 × {v('c', 100.0)})]"]
    elif k == "l_etrier":
        parts = [s, "p + 2·l_r", f"{v('p')} + 2 × {v('l_r', 1000.0)}"]
    elif k in ("n_cours", "L_d", "M_s"):
        noms = [x["symbole"] for x in f["termes"]]
        parts = [s, " + ".join(noms) if len(noms) <= 4 else f"Σ ({len(noms)} termes)", " + ".join(_v(x["valeur"]) for x in f["termes"])]
    elif k == "l_zone":
        parts = [s, "à − de", f"{v('à')} − {v('de')}"]
    elif k == "quotient":
        parts = [s, "l_z / s", f"{v('l_z')} / {v('s', 100.0)}"]
    elif k == "cours":
        parts = [s, "⌈q⌉", f"⌈{v('q')}⌉"]
    elif k == "n_zone":
        parts = [s, "n_c + fermeture", f"{v('n_c')} + {v('fermeture')}"]
    elif k == "A_c":
        e, w = ["b·h"], [f"{v('b')} × {v('h')}"]
        for ind in ("g", "d"):
            if f"b_{ind}" in t:
                e.append(f"b_{ind}·h_{ind}")
                w.append(f"{v('b_' + ind)} × {v('h_' + ind)}")
        parts = [s, " + ".join(e), " + ".join(w)]
    elif k == "V_c":
        parts = [s, "A_c·L", f"{v('A_c', 1e4)} m² × {v('L')}"]
    elif k == "M_c":
        parts = [s, "V_c·ρ_c", f"{v('V_c')} × {v('ρ_c')}"]
    elif k == "M_maj":
        parts = [s, "M_s·(1 + maj)", f"{v('M_s')} × (1 + {v('maj', 100.0)})"]
    elif k == "taux":
        parts = [s, "M_s,maj / V_c", f"{v('M_s,maj')} / {v('V_c')}"]
    elif k == "taux_net":
        parts = [s, "M_s / V_c", f"{v('M_s')} / {v('V_c')}"]
    elif k == "kg_m":
        parts = [s, "M_s,maj / L", f"{v('M_s,maj')} / {v('L')}"]
    else:                                                        # pragma: no cover — une formule nouvelle du moteur
        parts = [s, " · ".join(f"{x['symbole']} = {_v(x['valeur'])}" for x in f["termes"])]
    return {"parts": parts, "res": _res(f["resultat"], f["unite"]), "niveau": ""}


def _groupe(titre: str, formules: List[Dict], sous: str = "") -> str:
    if not formules:
        return ""
    st = f'<span class="ck-comb">{C.esc(sous)}</span>' if sous else ""
    return f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}{st}</div>{C.lignes_fx([_fx(f) for f in formules])}</div>'


# ------------------------------------------------------------------ le résultat
def _calculer(p: Dict) -> Dict:
    try:
        return MO.calculer(p)
    except ValueError as e:                                     # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(e)}


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE (l'accueil, la liste des notes) : une consultation, sans verdict ; le taux de la note
    (``moteur.synthese_note`` : Σ masses / Σ volumes) et sa poutre la plus armée."""
    ok = [(r["totaux"]["taux_majore"], i) for i, r in enumerate(resultats) if "totaux" in r]
    if not ok:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    t, i = max(ok)
    nom = etat["poutres"][i].get("nom") or f"Poutre {i + 1}"
    if len(ok) == 1:
        gouv = f"{nom} · {C.qte(t, 'kg/m³', 1)}"
    else:
        sn = MO.synthese_note([resultats[k] for _, k in ok])
        gouv = f"note : {C.qte(sn['taux_majore'], 'kg/m³', 1)} · la plus armée : {nom} ({C.qte(t, 'kg/m³', 1)})"
    return {"taux": None, "niveau": "i", "statut": C.STATUTS["i"][1], "gouvernant": gouv}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """Rien : les poutres de la note sont la LIGNE de l'élément et les rubriques de la colonne de gauche (``arbre``)."""
    return ""


def _verdict(r: Dict) -> str:
    t, b = r["totaux"], r["beton"]
    cmp_ = (f"{C.qte(t['masse_majoree'], 'kg', 1)} d'acier (+{C.NBSP}{C.nb(t['majoration'])}{C.NBSP}%) pour "
            f"{C.qte(b['volume'], 'm³', 3)} de béton<br><span class=\"tx-net\">sans majoration : {C.qte(t['taux'], 'kg/m³', 1)}</span>")
    return C.verdict_valeur(t["taux_majore"], "kg/m³", "Taux d'armature", cmp_, "c", 1, calcul="s0", classe="tx-verdict")


def _bloc_familles(r: Dict) -> str:
    rows = ""
    for f in r["familles"]:
        if not f["reperes"]:
            continue
        reps = ", ".join(str(x) for x in f["reperes"])
        cmp_ = f"{C.esc(' + '.join(f['libelles']))} · {C.qte(f['longueur'], 'm', 2)} · repère{'s' if len(f['reperes']) > 1 else ''} {reps}"
        rows += C.ligne_verif(f["libelle"], cmp_, _part(f["part"]), "c", calcul="s3" if f["cle"] == "etriers" else "s0",
                              piece=C.qte(f["masse"], "kg", 1))
    if not rows:
        rows = '<div class="hint">Aucun acier : ajouter un lit, un renfort, des barres de peau ou une zone d\'étriers.</div>'
    t = r["totaux"]
    ef = f"M<sub>s</sub> = {C.qte(t['masse'], 'kg', 1)} sans majoration · part de la masse d'acier"
    return E.bloc(0, "Acier par famille", "inc", C.qte(t["masse"], "kg", 1).replace(C.NBSP, " "), ef, rows, theme=True)


def _bloc_diametres(r: Dict) -> str:
    rows = ""
    for d in r["diametres"]:
        reps = ", ".join(str(x) for x in d["reperes"])
        cmp_ = f"{C.qte(d['longueur'], 'm', 2)} · {C.qte(d['masse_lineique'], 'kg/m', 3)} · repère{'s' if len(d['reperes']) > 1 else ''} {reps}"
        rows += C.ligne_verif(f"Ø {d['d']}", cmp_, _part(d["part"]), "c", calcul="s1", piece=C.qte(d["masse"], "kg", 1))
    ef = f"{len(r['diametres'])} diamètre{'s' if len(r['diametres']) > 1 else ''} · longueur totale {C.qte(r['totaux']['longueur'], 'm', 2)}"
    return E.bloc(1, "Par diamètre", "inc", f"{len(r['diametres'])} Ø", ef, rows, theme=True)


def _bloc_beton(r: Dict) -> str:
    b, sec, p = r["beton"], r["section"], r["poutre"]
    talons = " + talons" if sec["bg"] or sec["bd"] else ""
    rows = (C.ligne_verif("Section", f"A<sub>c</sub> = b·h{talons} · {C.esc(EN.LIBELLES_FORME[sec['forme']])}", None, "c", calcul="s2",
                          etat_txt=C.qte(b["aire_cm2"], "cm²", 0 if float(b["aire_cm2"]).is_integer() else 1).replace(C.NBSP, " "))
            + C.ligne_verif("Volume", f"V<sub>c</sub> = A<sub>c</sub>·L · L = {_m(p['L'])}", None, "c", calcul="s2",
                            etat_txt=C.qte(b["volume"], "m³", 3).replace(C.NBSP, " "))
            + C.ligne_verif("Masse", f"M<sub>c</sub> = V<sub>c</sub>·ρ<sub>c</sub> · ρ<sub>c</sub> = {C.qte(b['rho'], 'kg/m³', 0)}", None, "c",
                            calcul="s2", etat_txt=C.qte(b["masse"], "kg", 0).replace(C.NBSP, " ")))
    return E.bloc(2, "Béton", "inc", C.qte(b["volume"], "m³", 3).replace(C.NBSP, " "), f"{_dims(p)}", rows, theme=True)


def _bloc_note(etat: Dict, resultats: List[Dict]) -> str:
    """La NOTE entière (dès deux poutres) : Σ volumes, Σ masses, leur rapport (``moteur.synthese_note``)."""
    ok = [r for r in resultats if "totaux" in r]
    sn = MO.synthese_note(ok)
    rows = "".join(C.ligne_verif(p.get("nom") or f"P{i + 1}", f"{C.qte(r['totaux']['masse_majoree'], 'kg', 1)} · {C.qte(r['beton']['volume'], 'm³', 3)}",
                                 None, "c", etat_txt=C.qte(r["totaux"]["taux_majore"], "kg/m³", 1).replace(C.NBSP, " "))
                   for i, (p, r) in enumerate(zip(etat["poutres"], resultats)) if "totaux" in r)
    ef = (f"{sn['n']} poutres · {C.qte(sn['masse_majoree'], 'kg', 1)} d'acier majoré pour {C.qte(sn['volume'], 'm³', 3)} · "
          f"{C.qte(sn['longueur'], 'm', 2)} de poutres")
    return E.bloc(3, "Note entière", "inc", C.qte(sn["taux_majore"], "kg/m³", 1).replace(C.NBSP, " "), ef, rows, theme=True)


def _avertissements(r: Dict) -> str:
    av = sorted(r["avertissements"], key=lambda a: not a["grave"])
    return "".join(C.message(C.esc(a["message"]), a["grave"]) for a in av)


def _methode() -> str:
    return (f"Méthode : barres bornées aux abouts à l'enrobage (une filante mesure L − 2c), retours et recouvrements compris ; "
            f"cadres et épingles comptés hors tout à l'enrobage, deux retours de max({C.nb(MO.RETOUR_K)}{C.NBSP}Ø ; "
            f"{C.nb(MO.RETOUR_MIN)}{C.NBSP}mm), sans déduction des cintrages ; un étrier à chaque bout d'une suite de zones et un "
            f"tous les e au plus. Dans le doute, le côté qui commande le plus d'acier.")


# ------------------------------------------------------------------ les fenêtres de calcul
def _hyps(r: Dict) -> List[str]:
    p, t, rec = r["poutre"], r["totaux"], r["poutre"]["recouvrements"]
    o = _origine_txt(p)
    return [f"Poutre <b>{C.esc(p['nom'] or 'Poutre')}</b> · {C.esc(EN.LIBELLES_FORME[p['forme']])} · {_dims(p)} · c = {C.qte(p['enrobage'], 'cm', 1)}"
            + (f" · {C.esc(o)}" if o else ""),
            f"ρ<sub>s</sub> = {C.qte(p['rho_acier'], 'kg/m³', 0)} · ρ<sub>c</sub> = {C.qte(p['rho_beton'], 'kg/m³', 0)} · majoration "
            f"{C.nb(t['majoration'])}{C.NBSP}% (chutes et ligatures)",
            (f"recouvrements de {C.nb(rec['k'])}{C.NBSP}Ø au-delà de {C.qte(rec['l_com'], 'm', 0)}" if rec["actifs"] else "recouvrements non comptés")]


COLONNES_REPERES = ("Repère", "Longueur unitaire", "Longueur totale", "Masse", "Part")


def _volets(r: Dict, nom: str) -> str:
    hyps = _hyps(r)
    t = r["totaux"]
    sous = f"{_dims(r['poutre'])} · un quantitatif : aucune vérification"
    # 0 · les repères du bordereau et les totaux (le verdict et le bloc « Acier par famille » l'ouvrent)
    lignes = [{"nom": f"Repère {x['repere']}", "sous": C.esc(f"{x['designation']} · {x['position']} · {x['libelle']}"),
               "formule": f"l = {C.qte(x['l_unitaire'], 'm', 3)}", "valeur": f"{x['n']} × l = {C.qte(x['l_totale'], 'm', 2)}",
               "limite": C.qte(x["masse"], "kg", 2), "taux": _part(x["part"]), "niveau": "c"} for x in r["reperes"]]
    lignes.append({"nom": "Total", "sous": f"majoration {C.nb(t['majoration'])}{C.NBSP}% : {C.qte(t['masse_majoree'], 'kg', 2)}", "formule": "Σ repères",
                   "valeur": C.qte(t["longueur"], "m", 2), "limite": C.qte(t["masse"], "kg", 2), "taux": None, "niveau": "c",
                   "etat_txt": C.qte(t["taux_majore"], "kg/m³", 1), "gov": True})
    detail = _groupe("Totaux et taux", t["formules"])
    for x in r["reperes"]:
        detail += _groupe(f"Repère {x['repere']} · {x['designation']}", x["formules"], f"{x['libelle']} · {x['position']}")
    out = C.fenetre_calcul("s0", f"{nom} · repères et taux", sous, hyps, lignes, detail,
                           '<div class="ck-autres">Le taux est la masse d\'acier MAJORÉE (chutes et ligatures) rapportée au volume de béton '
                           'de la poutre, talons compris ; le ratio au mètre, la même masse rapportée à la longueur.</div>',
                           colonnes=COLONNES_REPERES)
    # 1 · par diamètre
    ld = [{"nom": f"Ø{C.NBSP}{d['d']}", "sous": f"repère{'s' if len(d['reperes']) > 1 else ''} {', '.join(str(k) for k in d['reperes'])}",
           "formule": "Σ L<sub>i</sub> · m", "valeur": f"{C.qte(d['longueur'], 'm', 2)} × {C.qte(d['masse_lineique'], 'kg/m', 3)}",
           "limite": C.qte(d["masse"], "kg", 2), "taux": _part(d["part"]), "niveau": "c"} for d in r["diametres"]]
    dd = "".join(_groupe(f"Ø {d['d']}", d["formules"]) for d in r["diametres"])
    out += C.fenetre_calcul("s1", f"{nom} · par diamètre", sous, hyps, ld, dd, "", colonnes=("Diamètre", "Formule", "Longueur × masse linéique", "Masse", "Part"))
    # 2 · le béton
    b = r["beton"]
    lb = [{"nom": "Section", "sous": C.esc(EN.LIBELLES_FORME[r["section"]["forme"]]), "formule": "A<sub>c</sub>", "valeur": C.qte(b["aire_cm2"], "cm²", 1),
           "limite": "—", "taux": None, "niveau": "c", "etat_txt": "béton"},
          {"nom": "Volume", "sous": f"L = {_m(r['poutre']['L'])}", "formule": "V<sub>c</sub> = A<sub>c</sub>·L", "valeur": C.qte(b["volume"], "m³", 3),
           "limite": "—", "taux": None, "niveau": "c", "etat_txt": "béton"},
          {"nom": "Masse", "sous": f"ρ<sub>c</sub> = {C.qte(b['rho'], 'kg/m³', 0)}", "formule": "M<sub>c</sub> = V<sub>c</sub>·ρ<sub>c</sub>",
           "valeur": C.qte(b["masse"], "kg", 0), "limite": "—", "taux": None, "niveau": "c", "etat_txt": "béton"}]
    out += C.fenetre_calcul("s2", f"{nom} · béton", sous, hyps, lb, _groupe("Béton", b["formules"]),
                            '<div class="ck-autres">Les talons sont dans la section (âme et talons) : leur béton est compté.</div>',
                            colonnes=("Grandeur", "Formule", "Valeur", "Limite", "Nature"))
    # 3 · les étriers par zone, et la longueur d'un cours de chaque type
    lz = [{"nom": z["nom"], "sous": C.esc(f"de {_m(z['de'])} à {_m(z['a'])} · e = {_cm(z['s'])} cm · {z['type']}"),
           "formule": "⌈l<sub>z</sub>/e⌉" + (" + 1 de fermeture" if z["fermeture"] else ""),
           "valeur": f"{C.esc(_v(z['longueur']))}/{C.esc(_v(z['s'] / 100.0))} = {C.esc(_v(z['quotient']))}", "limite": f"{z['cours']} cours",
           "taux": None, "niveau": "c", "etat_txt": f"{z['n']} étrier{'s' if z['n'] > 1 else ''}"} for z in r["zones"]]
    dz = "".join(_groupe(f"Zone {z['nom']}", z["formules"], f"suite {z['suite'] + 1}") for z in r["zones"])
    for ty in r["types"]:
        for j, li in enumerate(ty["lignes"]):
            dz += _groupe(f"Type {ty['nom']} · ligne {j + 1} · {EN.LIBELLES_LIGNE[li['forme']]}", li["formules"])
        if ty["talon"]:
            dz += _groupe(f"Type {ty['nom']} · cadre de talon", ty["talon"]["formules"])
    n = sum(z["n"] for z in r["zones"])
    out += C.fenetre_calcul("s3", f"{nom} · étriers par zone", sous, hyps, lz, dz,
                            f'<div class="ck-autres">{n} étriers : ⌈l<sub>z</sub>/e − 10<sup>−9</sup>⌉ cours par zone (autant d\'intervalles de e au '
                            'plus qu\'il en faut), et UN cours de fermeture par suite de zones jointives — deux zones qui se touchent ne comptent '
                            'pas deux fois l\'étrier de leur jonction.</div>',
                            colonnes=("Zone", "Règle", "l<sub>z</sub>/e", "Cours", "Étriers"))
    return out


# ------------------------------------------------------------------ les vues
def _entete(r: Dict, titre: str, legende: str) -> str:
    p, t = r["poutre"], r["totaux"]
    sous = f"{C.esc(p['nom'] or 'Poutre')} · {C.esc(EN.LIBELLES_FORME[p['forme']])} · {_dims(p)} · c{C.NBSP}={C.NBSP}{C.qte(p['enrobage'], 'cm', 1)}"
    lignes = []
    o = _origine_txt(p)
    if o:
        lignes.append(o[0].upper() + o[1:])
    lignes.append(f"{len(r['reperes'])} repères · acier {C.fr(t['masse_majoree'], 1)} kg majoré de {C.nb(t['majoration'])} % · béton "
                  f"{C.fr(r['beton']['volume'], 3)} m³")
    bande = [("T", "", t["taux_majore"], "kg/m³"), ("M", "s", t["masse_majoree"], "kg"), ("m", "L", t["kg_m_majore"], "kg/m")]
    return E.entete(titre, sous, lignes, bande, legende)


def _vues(r: Dict, ident: str) -> List[Dict]:
    out = []
    for v in DT.vues(r, ident):
        v["entete"] = _entete(r, v["titre"].upper(), v["legende"])
        out.append(v)
    return out


def resultat(etat: Dict) -> Dict:
    """Le résultat de la poutre ouverte : la tête (le nom, les dimensions), la provenance d'une poutre reçue, l'avis
    d'ouverture d'une ancienne note et ce qui a suivi la dernière saisie (dits une fois, retirés de l'état), le
    verdict de valeur, les blocs, les avertissements, le pied ; les fenêtres de calcul, les vues, la colonne de
    gauche, le résumé de la note."""
    avis = [AVIS[k] for k in (etat.pop(CLE_AVIS, None),) if k in AVIS]
    suivis = etat.pop(CLE_SUIVI, None) or []
    p = _poutre(etat)
    r = MO.calculer(p)
    resultats = [r if i == etat["actif"] else _calculer(q) for i, q in enumerate(etat["poutres"])]
    nom = C.esc(p.get("nom") or "Poutre")
    forme = "" if p["forme"] == "rectangulaire" else f" · {C.esc(EN.LIBELLES_FORME[p['forme']])}"
    tete = E.tete_resultats(nom, _dims(p) + forme)
    o = _origine_txt(p)
    msgs = "".join(C.message(C.esc(a)) for a in avis)
    if o:
        msgs += C.message(f"<b>Poutre {C.esc(o)}</b> (module « Poutres béton armé ») : ses sections sont devenues les tronçons ; "
                          "régler ici où s'arrêtent les barres, les zones d'étriers, les retours.")
    if suivis:
        msgs += C.message("Ont suivi : " + C.esc(" ; ".join(suivis)) + ".")
    blocs = _bloc_familles(r) + _bloc_diametres(r) + _bloc_beton(r) + (_bloc_note(etat, resultats) if len(etat["poutres"]) > 1 else "")
    t = r["totaux"]
    pied = (f'<div class="pfoot tx-pied"><div class="mass"><span>Par mètre de poutre</span><b>{C.qte(t["kg_m_majore"], "kg/m", 1)}</b></div>'
            f'{C.bouton_telecharger("bordereau", "Bordereau CSV")}</div>')
    methode = f'<div class="et-hyps">{C.aide(C.esc(_methode()))}</div>'
    corps = (f'<div class="pbody">{tete}{msgs}{_verdict(r)}<div class="et-blocs">{blocs}</div>{_avertissements(r)}{methode}</div>{pied}')
    return {"resultat": corps, **({"avis": avis} if avis else {}), "volets": _volets(r, nom), "vues": _vues(r, f"tx{etat['actif']}"),
            "phrase": "", "statut_html": C.statut("i"), "resume": resume_note(etat, resultats), "donnees": arbre(etat, r),
            "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le bordereau CSV
def telecharger(etat: Dict, quoi: str) -> Optional[Dict]:
    """« bordereau » : le bordereau de la poutre OUVERTE en CSV (``moteur.bordereau_csv`` : « ; », virgule décimale, les
    repères, la synthèse par Ø, le total), sous un nom sûr tiré du nom de la poutre. Tout autre ``quoi`` : None."""
    if quoi != "bordereau":
        return None
    p = _poutre(etat)
    return {"nom": C.nom_fichier(f"Bordereau {p.get('nom') or 'poutre'}", "csv"), "type": "text/csv",
            "texte": MO.bordereau_csv(MO.calculer(p))}

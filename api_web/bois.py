# -*- coding: utf-8 -*-
"""Le module « Poutres bois » du site, refait le 10/11/2026 sur l'Eurocode 5 (docs/site/LOT_Q_QUATRE_MODULES.md § 1 ; le
moteur : ``bois/poutre``, docs/bois/POUTRE_BOIS_MOTEUR.md), sur l'ÉTABLI COMPACT — le format de la poutre béton armé
et des cornières (api_web/etabli_compact.py, js/compact.js) :

- à gauche, la ligne de la poutre ouverte (son nom, la liste des poutres de la note, copier, supprimer) puis ses quatre
  rubriques : 1 Poutre (le bois — massif puis lamellé-collé —, la section standard, qui ouvre le SÉLECTEUR, ou sur
  mesure, la portée), 2 Charges (kN/m, ou kN/m² × entraxe ; G hors poids propre, Q et sa NATURE, qui fixe sa durée et
  ψ₂ ; le poids propre compté), 3 Appuis et maintien (la longueur d'appui, le débord, le maintien latéral),
  4 Hypothèses (secondaire : la classe de service, γ_G et γ_Q, la durée et ψ₂ — verrouillés hors de la nature
  « autre » —, l'effet système, les limites de flèche, la contre-flèche) ; données condensées, les explications dans
  les ⓘ ;
- au centre, trois VUES sous leur en-tête technique (api_web/dessin_bois.py) : le schéma et les efforts de la
  combinaison qui gouverne, la déformée, la section ;
- à droite, la tête (le nom ; « 75 × 225 · C24 · L 4,00 m »), le VERDICT (le taux, la vérification qui gouverne, sa
  comparaison « valeur ≤ limite », sa combinaison), deux blocs THÉMATIQUES — « ELU · résistance » (flexion, dont la
  pièce est la section et ouvre le sélecteur ; cisaillement ; appui) et « ELS · flèches » (instantanée, finale nette)
  —, chaque ligne ouvrant SA fenêtre de calcul et le bouton du bloc « tout le calcul » ; les avertissements, les
  hypothèses, ce qui n'est pas couvert ; au pied la masse, le poids propre, la légende ;
- les FENÊTRES DE CALCUL : les hypothèses en pastilles, le tableau (une ligne par combinaison), puis le détail groupé
  — chaque ligne « symbole = expression = valeurs = résultat » est le gabarit du MOTEUR où chaque terme est remplacé
  par sa valeur mise en forme (le test la réévalue) ;
- le SÉLECTEUR « section » : les sections standard du type du bois, chacune recalculée avec les charges, triées par
  masse — ✓ la retenue, ★ la plus légère qui tient.

Ce fichier MET EN FORME : il ne calcule rien et n'écrit aucune formule ; il CHOISIT parmi ce que le moteur a nommé et
COMPOSE les phrases. Toute saisie passe par ``moteur.normaliser`` : hors domaine, elle est REFUSÉE avec le message du
moteur et la poutre reste intacte. Une note du module d'avant (version 1, la copie de l'écran Streamlit) se rouvre sur
des poutres NEUVES qui gardent leurs noms, et l'AVIS le dit une fois à l'ouverture (CLE_AVIS, le mécanisme des
cornières : la clé est retirée de l'état dès que l'avis est rendu).
"""
import copy
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from bois.poutre import donnees as D
from bois.poutre import entree as EN
from bois.poutre import moteur as MO

from . import commun as C
from . import dessin_bois as DB
from . import etabli_compact as E

MODULE = "bois"
PAGE = "poutre-bois"
VERSION = 2                     # l'état de la note : {"version": 2, "poutres": [...], "actif": 0} (1 : la copie de Streamlit)
NOM_DEFAUT = "P1"               # « P1 », « P2 »… : court, comme « V1 » de la poutre béton armé
# L'AVIS dit UNE fois à l'ouverture : ``normaliser`` (une note d'avant) ou ``modifier`` (un changement de type de bois)
# le posent sous CLE_AVIS ; ``resultat`` le montre en tête du résultat et le RETIRE de l'état.
CLE_AVIS = "avis_ouverture"
AVIS_V1 = ("Note d'avant le 10/11/2026 : le module « Poutres bois » a été refait sur l'Eurocode 5, les données sont à ressaisir "
           "(les noms des poutres sont gardés).")
AVIS_NOM = f" Un nom de plus de {D.NOM_MAX} caractères a été coupé."
CLES_AVIS = ("v1", "v1_nom", "section")

SEGMENTS_SECTION = (("standard", "Standard"), ("mesure", "Sur mesure"))
SEGMENTS_MODE = (("lineaire", "kN/m"), ("surfacique", "kN/m²"))
SEGMENTS_MAINTIEN = (("continu", "Continu"), ("appuis", "Aux appuis"), ("entretoises", "Entretoises"))
GROUPES_CLASSES = {D.MASSIF: "Bois massif résineux · EN 338", D.LAMELLE: "Lamellé-collé résineux · EN 14080"}
# le nom COURT d'une vérification (1ʳᵉ ligne d'un résultat, sans coupure dans la colonne)
COURTS = {"flexion": "Flexion", "cisaillement": "Cisaillement", "appui": "Appui", "fleche_inst": "Flèche instantanée",
          "fleche_fin": "Flèche finale nette"}
GROUPES = ((MO.ELU, "ELU · résistance"), (MO.ELS, "ELS · flèches"))
NOMBRES = tuple(D.BORNES)                                       # les champs numériques (leurs bornes : le moteur)
CASES = ("charges/poids_propre", "appui/verifier", "systeme")
ENTIERS = ("service", "lim_inst", "lim_fin")
CHOIX = ("charges/mode", "charges/nature", "charges/duree", "maintien/type")

# une note du module d'avant (version 1, la copie de l'écran Streamlit), telle qu'un fichier .stab d'avant le 10/11/2026
# la porte : le parcours la rejoue (la relecture, l'avis), tests/test_api_web_bois.py la contrôle
NOTE_V1 = {"version": 1, "actif": 1,
           "poutres": [{"nom": "Solive du séjour", "classe": "C24", "service": 1, "duree": "long", "standard": True, "section": "38x150",
                        "b": 80.0, "h": 240.0, "L": 4.0, "G": 2.0, "Q": 3.0, "gamma_G": 1.35, "gamma_Q": 1.5, "psi2": 0.3,
                        "lim_inst": "L/300", "lim_fin": "L/200", "appui": False, "a": 60.0},
                       {"nom": "Chevêtre", "classe": "C30", "service": 2, "duree": "moyen", "standard": False, "section": "38x150",
                        "b": 100.0, "h": 250.0, "L": 2.5, "G": 4.0, "Q": 5.0, "gamma_G": 1.35, "gamma_Q": 1.5, "psi2": 0.3,
                        "lim_inst": "L/300", "lim_fin": "L/200", "appui": True, "a": 80.0}]}


def _note(**champs) -> Dict:
    """Une note neuve dont la poutre reçoit ``champs`` (« charges__nature » écrit dans le groupe « charges »)."""
    p = MO.nouvelle_poutre(NOM_DEFAUT)
    for cle, val in champs.items():
        if "__" in cle:
            g, c = cle.split("__")
            p[g][c] = val
        else:
            p[cle] = val
    return {"version": VERSION, "poutres": [p], "actif": 0}


# les requêtes de plus pour le parcours (api_web/parcours.py) et le relevé du paquet : une saisie refusée, une section
# sur mesure, un lamellé-collé, le déversement (maintien aux appuis, entretoises trop espacées : refusées), la nature
# « autre » et sa durée, la saisie surfacique, l'appui retiré, la contre-flèche, le sélecteur, la note d'avant
PARCOURS = [
    {"action": "modifier", "chemin": ["L"], "valeur": "0,2"},
    {"action": "modifier", "etat": _note(standard=False), "chemin": ["b"], "valeur": "62,5"},
    {"action": "modifier", "chemin": ["classe"], "valeur": "GL28c"},
    {"action": "modifier", "etat": _note(section="45x245", b=45.0, h=245.0, L=5.0), "chemin": ["maintien", "type"], "valeur": "appuis"},
    {"action": "modifier", "etat": _note(maintien__type="entretoises"), "chemin": ["maintien", "entraxe"], "valeur": "9"},
    {"action": "modifier", "etat": _note(charges__nature="autre"), "chemin": ["charges", "duree"], "valeur": "instantane"},
    {"action": "modifier", "etat": _note(charges__nature="autre"), "chemin": ["charges", "psi2"], "valeur": "0,5"},
    {"action": "modifier", "etat": _note(charges__mode="surfacique"), "chemin": ["charges", "entraxe"], "valeur": "0,6"},
    {"action": "modifier", "chemin": ["charges", "psi2"], "valeur": "0,7"},
    {"action": "modifier", "chemin": ["contre_fleche"], "valeur": "20"},
    {"action": "modifier", "chemin": ["charges", "G"], "valeur": "3"},
    {"action": "choisir", "role": "section", "nom": "45 × 220"},
    {"action": "calculer", "etat": NOTE_V1},
]


# ------------------------------------------------------------------ l'état de la note : des poutres
def nouvel_etat() -> Dict:
    """Une note neuve : une poutre aux défauts de la conception § 1.5 (C24, 75 × 225, L = 4,00 m, G = 0,6 et Q = 0,8
    kN/m en habitation, classe de service 1, appui de 100 mm, maintien continu) — conforme."""
    return {"version": VERSION, "poutres": [MO.nouvelle_poutre(NOM_DEFAUT)], "actif": 0}


def _indice(x) -> int:
    """L'indice de la poutre ouverte, lu sans faire tomber la note : un nombre entier fini, sinon la première."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return 0
    return int(x)


def _depuis_ancienne(etat: Dict, poutres: List) -> Dict:
    """Une note du module d'AVANT (version 1, ou illisible comme telle) : les NOMS sont gardés, les poutres sont
    neuves — les notes de l'ancien module étaient des essais, rien n'est converti (conception, en tête) — et l'avis
    d'ouverture est posé. Un nom trop long est coupé, et l'avis le dit."""
    out = {"version": VERSION, "poutres": []}
    coupe = False
    for i, p in enumerate(poutres):
        nom = p.get("nom") if isinstance(p, dict) else None
        nom = nom.strip() if isinstance(nom, str) and nom.strip() else f"P{i + 1}"
        if len(nom) > D.NOM_MAX:
            nom, coupe = nom[:D.NOM_MAX], True
        out["poutres"].append(MO.nouvelle_poutre(nom))
    out["actif"] = min(max(0, _indice(etat.get("actif"))), len(out["poutres"]) - 1)
    out[CLE_AVIS] = "v1_nom" if coupe else "v1"
    return out


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu par le moteur ; une poutre illisible lève une erreur claire
    (le routeur la rend, l'écran l'affiche). Une note sans poutre est une note neuve ; une liste qui n'en est pas une
    est une note abîmée (refusée). Une note qui n'est pas de la version d'aujourd'hui est une note du module d'AVANT
    (``_depuis_ancienne``) ; d'une version plus récente, elle est refusée (rien n'est deviné)."""
    poutres = etat.get("poutres") if isinstance(etat, dict) else None
    if not poutres:
        return nouvel_etat()
    if not isinstance(poutres, list):
        raise ValueError("note abîmée : « poutres » doit être une liste")
    version = etat.get("version")
    if version != VERSION:
        if isinstance(version, (int, float)) and not isinstance(version, bool) and version > VERSION:
            raise ValueError(f"note enregistrée par une version plus récente du module (version {version}) : elle ne se relit pas ici")
        return _depuis_ancienne(etat, poutres)
    out = {"version": VERSION, "poutres": []}
    for i, p in enumerate(poutres):
        if not isinstance(p, dict):
            raise ValueError(f"poutre n° {i + 1} illisible : un groupe de champs est attendu")
        try:
            out["poutres"].append(MO.normaliser(p))
        except ValueError as e:
            raise ValueError(f"poutre « {p.get('nom', i + 1)} » : {e}") from None
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
    """Les poutres de la note : en ouvrir une, en ajouter, en copier une (``index`` ; sans : l'ouverte — la copie
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
        p["nom"] = f"{p.get('nom') or 'Poutre'} (copie)"[:D.NOM_MAX]
        ps.insert(i + 1, p)
        etat["actif"] = i + 1
    elif action == "supprimer" and len(ps) > 1:
        i = min(max(0, _indice(index)), len(ps) - 1)
        del ps[i]
        if etat["actif"] > i or etat["actif"] >= len(ps):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
def _changer_classe(etat: Dict, p: Dict, classe) -> None:
    """Le bois change. Dans le même TYPE, rien d'autre ne bouge. D'un type à l'autre (massif ↔ lamellé-collé), la
    section GARDE ses dimensions : standard si le nouveau type la propose aussi, sinon « sur mesure » — et l'avis le
    dit (aucune dimension n'est changée en silence). La section standard en réserve devient la plus proche du type."""
    avant = EN.type_classe(p["classe"])
    apres = EN.type_classe(classe if isinstance(classe, str) else str(classe))
    p["classe"] = classe
    if apres == avant:
        return
    b, h = p["b"], p["h"]
    meme = EN.tag(b, h)
    if meme in [s["nom"] for s in EN.sections(apres)]:
        p["section"] = meme
    else:
        if p["standard"]:
            p["standard"] = False
            etat[CLE_AVIS] = "section"
        p["section"] = EN.section_proche(apres, b, h)
    if p["standard"]:
        p["b"] = p["h"] = None


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la poutre ouverte ; ``chemin`` : « L », « charges/G », « appui/a », « maintien/type »…
    Le moteur relit la poutre : une valeur hors de son domaine est REFUSÉE avec son message (jamais ramenée en
    silence), et la poutre reste intacte. Les effets de bord de la saisie : « standard » passe de la section standard
    à ses dimensions sur mesure, et retour (la section standard la plus proche) ; le bois qui change de type garde la
    section (``_changer_classe``) ; la nature de Q fixe sa durée et ψ₂ (sauf « autre », qui garde les derniers)."""
    p = _poutre(etat)
    avant, avis_avant = copy.deepcopy(p), etat.get(CLE_AVIS)
    cle = "/".join(str(c) for c in chemin)
    groupe, _, champ = cle.rpartition("/")
    cible = p[groupe] if groupe in EN.CLES_GROUPES else p
    try:
        if cle == "nom":
            nom = str(valeur if valeur is not None else "").strip()
            if len(nom) > D.NOM_MAX:
                return f"Nom trop long : {len(nom)} caractères (au plus {D.NOM_MAX})"
            p["nom"] = nom
        elif cle == "classe":
            _changer_classe(etat, p, valeur)
        elif cle == "standard":
            v = {"standard": True, "mesure": False, True: True, False: False}.get(valeur)
            if v is None:
                return f"Section : « {valeur} » (standard ou sur mesure)"
            if v and not p["standard"]:
                p["section"] = EN.section_proche(EN.type_classe(p["classe"]), p["b"], p["h"])
                p["b"] = p["h"] = None
            p["standard"] = v
        elif cle == "section":
            p.update(standard=True, section=EN.lire_tag(valeur), b=None, h=None)
        elif cle in NOMBRES:
            x = C.lire_nombre(valeur)
            if x is None:
                return f"Nombre illisible : « {valeur} »"
            cible[champ] = x
        elif cle in CASES:
            if not isinstance(valeur, bool):
                return f"Case à cocher : vrai ou faux est attendu (reçu « {valeur} »)"
            cible[champ] = valeur
        elif cle in ENTIERS:
            x = C.lire_nombre(valeur)
            if x is None or not float(x).is_integer():
                return f"Choix illisible : « {valeur} »"
            p[cle] = int(x)
        elif cle in CHOIX:
            cible[champ] = valeur
            if cle == "charges/nature" and valeur != D.NATURE_LIBRE:
                cible["duree"] = cible["psi2"] = None           # déduits de la nature (le moteur les remplit)
        else:
            return f"Champ inconnu : « {cle} »"
        etat["poutres"][etat["actif"]] = MO.normaliser(p)
    except (ValueError, KeyError, TypeError) as e:
        etat["poutres"][etat["actif"]] = avant
        if avis_avant is None:
            etat.pop(CLE_AVIS, None)
        texte = str(e)
        return f"Saisie refusée : {texte[:1].lower() + texte[1:]}"
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le clic dans le sélecteur : la section standard ``nom`` (« 45 × 220 ») est retenue (la section repasse en
    « standard ») ; une section qui n'est pas dans la liste du type du bois lève l'erreur du moteur."""
    p = copy.deepcopy(_poutre(etat))
    p.update(standard=True, section=EN.lire_tag(nom), b=None, h=None)
    etat["poutres"][etat["actif"]] = MO.normaliser(p)


# ------------------------------------------------------------------ les nombres
_ACCOLADES = re.compile(r"\{([^{}]+)\}")
DECIMALES = {"N/mm²": 2, "kN": 2, "kN·m": 2, "kN/m": 3, "kN/m²": 2, "mm²": 0, "mm³": 0, "mm⁴": 0, "kg/m": 2, "kg": 1, "m": 2}


def _nd(x: float, nd: int, sig: int = 2) -> int:
    """Assez de décimales pour garder ``sig`` chiffres significatifs aux petites valeurs (0,0345 kN·m, pas « 0,03 »)."""
    a = abs(x)
    while a and nd < 6 and a < 10 ** (sig - 1 - nd):
        nd += 1
    return nd


def _n(x: Optional[float], unite: str = "") -> str:
    """Un RÉSULTAT à la précision d'usage : contraintes et efforts au centième, charges au millième, flèches au
    centième de millimètre (les longueurs en mm entières), sections entières, coefficients à trois décimales — et
    toujours deux chiffres significatifs."""
    if x is None:
        return "—"
    if unite == "mm":
        return C.fr(x, _nd(x, 2)) if abs(x) < 100 else C.fr(x, 0 if float(x).is_integer() else 1)
    if unite == "":
        return C.nb(round(x, _nd(x, 3)), _nd(x, 3))
    nd = DECIMALES.get(unite, 2)
    return C.fr(x, _nd(x, nd) if nd else 0)


def _nu(x: Optional[float], unite: str) -> str:
    """Un résultat suivi de son unité (insécable)."""
    if x is None:
        return "—"
    return f"{_n(x, unite)}{C.NBSP}{unite}" if unite else _n(x, unite)


def _v(x: float) -> str:
    """Une VALEUR substituée dans une formule, écrite sans la dénaturer : 75 reste 75, 1,35 reste 1,35, 4 000 et
    11 000 gardent leurs chiffres ; une grandeur calculée longue (71 191 406,25 ; 0,669528…) est arrondie à quatre
    chiffres significatifs au moins ; un nombre négatif est mis entre parenthèses (il suit un signe)."""
    a = abs(x)
    if a >= 1000:
        s = C.fr(x, 0 if a >= 1e5 or float(x).is_integer() else 1)
    elif a >= 100:
        s = C.nb(round(x, 1), 1)
    elif a >= 10:
        s = C.nb(round(x, 2), 2)
    elif a >= 1:
        s = C.nb(round(x, 3), 3)
    else:
        nd = _nd(x, 4, 4)
        s = C.nb(round(x, nd), nd)
    return f"({s})" if x < 0 else s


def _pct(t: Optional[float]) -> str:
    """Un taux EXACT (le détail du calcul), au dixième, dans la bande de sa couleur."""
    return f"{C.pc(C.pct(t, 1))}{C.NBSP}%" if t is not None else "—"


def _symb(s: str) -> str:
    return C.symb_math(s)


def _niv(v: Dict) -> str:
    """La couleur d'une vérification (ou d'une combinaison) : rouge dès que le moteur la dit non conforme, sinon son taux."""
    return "r" if v["etat"] == MO.NOK else C.niveau(v["taux"])


def _niv_res(r: Dict) -> str:
    return "r" if r["statut"] == MO.NOK else C.niveau(r["taux_max"])


def _section_txt(geo: Dict) -> str:
    """« 75 × 225 » : la section, telle que le moteur la désigne (texte échappé)."""
    return C.esc(geo["designation"])


def _comb(r: Dict, cle: Optional[str]) -> Optional[Dict]:
    return next((c for c in r["combinaisons"] if c["cle"] == cle), None)


def _comb_txt(r: Dict, v: Dict, court: bool = False) -> str:
    """La combinaison qui gouverne une vérification, en HTML : « G + Q · moyen terme » ; à l'ELS, « ELS caractéristique »."""
    c = _comb(r, v["combinaison"])
    if c is None:
        return "ELS caractéristique" if not court else "ELS"
    return C.esc(c["nom"]) if court else f"{C.esc(c['nom'])} · {C.esc(c['libelle_duree'])}"


# ------------------------------------------------------------------ le cadre Données : les quatre rubriques
def _menu_classes(p: Dict) -> str:
    """Le menu des classes de résistance, groupé : d'abord le bois massif (C14 à C50), puis le lamellé-collé (GL…)."""
    groupes = ""
    for typ in D.TYPES:
        opts = "".join(f'<option value="{C.esc(c)}"{" selected" if c == p["classe"] else ""}>{C.esc(c)} · {C.esc(D.COURTS_TYPE[typ])}</option>'
                       for c, m in D.CLASSES.items() if m["type"] == typ)
        groupes += f'<optgroup label="{C.esc(GROUPES_CLASSES[typ])}">{opts}</optgroup>'
    return (f'<div class="field select"><select data-chemin="{C._attr_chemin(["classe"])}" data-sorte="choix">{groupes}</select>'
            f'{C.icon("chevron-down")}</div>')


def _bouton_section(r: Dict) -> str:
    """La section standard retenue, en bouton : il ouvre le SÉLECTEUR (les sections du type du bois, recalculées)."""
    return (f'<button type="button" class="bo-pk" data-pk="section" title="Choisir une section standard : la liste triée par masse et '
            f'calculée avec vos charges (✓ la retenue, ★ la plus légère qui tient)"><span>{_section_txt(r["geometrie"])}{C.NBSP}mm</span>'
            f'{C.icon("chevron-down")}</button>')


def _rubrique_poutre(p: Dict, r: Dict) -> str:
    mat, geo, co = r["materiau"], r["geometrie"], r["coefficients"]
    aide_bois = (f"{p['classe']} ({mat['norme']}) : f_m,k {C.nb(mat['fm_k'])} · f_v,k {C.nb(mat['fv_k'])} · f_c,90,k {C.nb(mat['fc90_k'])} · "
                 f"E_0,mean {C.fr(mat['E0_mean'], 0)} · E_0,05 {C.fr(mat['E0_05'], 0)} · G_mean {C.nb(mat['G_mean'])} N/mm² · "
                 f"ρ_mean {C.nb(mat['rho_mean'])} kg/m³. {mat['libelle_type'].capitalize()} : γ_M = {C.nb(co['gamma_M'])} (annexe belge).")
    s = C.ligne("Bois", _menu_classes(p), "", aide_bois, full=True)
    s += C.ligne("Section", C.segments(["standard"], "standard" if p["standard"] else "mesure", SEGMENTS_SECTION), full=True)
    if p["standard"]:
        s += C.ligne("b × h", _bouton_section(r), "", f"Une section standard du {D.COURTS_TYPE[mat['type']]} : la liste s'ouvre, chaque section "
                     "y est recalculée avec vos charges (taux, ✓ la retenue, ★ la plus légère qui tient).", full=True)
    else:
        s += C.ligne2(("b", C.num(["b"], p["b"], "mm"), "La largeur de la section."),
                      ("h", C.num(["h"], p["h"], "mm"), "La hauteur de la section, dans le plan de flexion."))
    s += C.ligne("portée", C.num(["L"], p["L"], "m"), "L", "La portée entre AXES des appuis ; poutre sur deux appuis simples.")
    s += C.aide(f"A = {C.qte(geo['A'] / 100, 'cm²', 1)} · W = {C.qte(geo['W'] / 1e3, 'cm³', 1)} · I = {C.qte(geo['I'] / 1e4, 'cm⁴', 0)} · "
                f"{C.qte(r['masse_lineique'], 'kg/m', 2)}")
    return s


def _rubrique_charges(p: Dict, r: Dict) -> str:
    ch, rc, co = p["charges"], r["charges"], r["coefficients"]
    s = C.ligne("Saisie en", C.segments(["charges", "mode"], ch["mode"], SEGMENTS_MODE), "",
                "kN/m : les charges linéaires que porte la poutre. kN/m² : les charges du plancher, multipliées par l'entraxe des "
                "pièces (gîtes, chevrons).", full=True)
    if ch["mode"] == "surfacique":
        s += C.ligne2(("g<sub>s</sub>", C.num(["charges", "g"], ch["g"], "kN/m²"), "La charge permanente du plancher, hors poids propre de la pièce."),
                      ("q<sub>s</sub>", C.num(["charges", "q"], ch["q"], "kN/m²"), "La charge variable du plancher."))
        s += C.ligne("entraxe", C.num(["charges", "entraxe"], ch["entraxe"], "m"), "e", "L'entraxe des pièces : G = g_s·e, Q = q_s·e.")
        s += C.aide(f"G = {C.qte(rc['G'], 'kN/m', 3)} · Q = {C.qte(rc['Q'], 'kN/m', 3)}")
    else:
        s += C.ligne2(("G", C.num(["charges", "G"], ch["G"], "kN/m"), "La charge permanente, HORS poids propre de la poutre (compté à part, ci-dessous)."),
                      ("Q", C.num(["charges", "Q"], ch["Q"], "kN/m"), "La charge variable ; sa nature fixe sa durée (k_mod) et ψ₂ (fluage)."))
    s += C.ligne("Nature Q", C.choix(["charges", "nature"], ch["nature"], [(k, D.LIBELLES_NATURE[k]) for k in D.CLES_NATURE]), "",
                 "La nature de la charge variable fixe sa classe de durée (EN 1995-1-1 tableau 2.2 et annexe belge), donc le k_mod de la "
                 "combinaison G + Q, et ψ₂ (NBN EN 1990 ANB), donc la part de Q qui flue. « Autre » : durée et ψ₂ se saisissent dans les "
                 "Hypothèses.", full=True)
    libre = " · à saisir dans les Hypothèses" if co["nature_libre"] else ""
    s += C.aide(f"{C.esc(co['libelle_duree'])} ({C.esc(D.PERIODES_DUREE[co['duree']])}) · k<sub>mod</sub> = {C.nb(co['kmod_variable'])} · "
                f"ψ<sub>2</sub> = {C.nb(co['psi2'])}{libre}")
    s += C.case(["charges", "poids_propre"], ch["poids_propre"], f"Poids propre compté · {C.qte(rc['poids_propre'], 'kN/m', 3)}")
    return s


def _rubrique_appuis(p: Dict, r: Dict) -> str:
    ap, mt, dv = p["appui"], p["maintien"], r["deversement"]
    s = C.ligne2(("a appui", C.num(["appui", "a"], ap["a"], "mm"),
                  "La longueur d'appui, comptée le long de la poutre. La compression perpendiculaire au fil se vérifie sur la largeur b et "
                  "la longueur efficace l_ef,a = a + 30 mm côté travée + 30 mm côté about (sans dépasser a, ni le débord)."),
                 ("débord", C.num(["appui", "debord"], ap["debord"], "mm"),
                  "La longueur de poutre au-delà de l'appui, côté about (0 : la poutre s'arrête au bord de l'appui). Elle augmente la "
                  "longueur efficace d'appui, de 30 mm au plus."))
    s += C.case(["appui", "verifier"], ap["verifier"], "Vérifier la compression à l'appui")
    s += C.ligne("Maintien", C.segments(["maintien", "type"], mt["type"], SEGMENTS_MAINTIEN), "",
                 "Le maintien latéral de la fibre comprimée (le dessus de la poutre). Continu : un plancher ou un platelage fixé, pas de "
                 "déversement (k_crit = 1). Aux appuis : la poutre est libre entre ses appuis, l_ef = 0,9·L + 2·h. Entretoises : l_ef = "
                 "leur entraxe + 2·h. Dans les trois cas, les appuis sont supposés maintenus en torsion (EN 1995-1-1 tableau 6.1).", full=True)
    if mt["type"] == "entretoises":
        s += C.ligne("entraxe", C.num(["maintien", "entraxe"], mt["entraxe"], "m"), "e<sub>m</sub>",
                     "L'entraxe des entretoises qui maintiennent la fibre comprimée (le plus grand, s'ils diffèrent) ; au plus la portée.")
    if dv["l_ef"] is None:
        s += C.aide(f"k<sub>crit</sub> = {_n(dv['kcrit'])} · pas de déversement")
    else:
        s += C.aide(f"l<sub>ef</sub> = {C.qte(dv['l_ef'], 'mm', 0)} · λ<sub>rel,m</sub> = {_n(dv['lambda_rel'])} · k<sub>crit</sub> = {_n(dv['kcrit'])}")
    return s


def _rubrique_hypotheses(p: Dict, r: Dict) -> str:
    co, ch = r["coefficients"], p["charges"]
    libre = co["nature_libre"]
    s = C.ligne("Service", C.segments(["service"], str(p["service"]), [(str(k), str(k)) for k in D.CLASSES_SERVICE]), "",
                "La classe de service (EN 1995-1-1 §2.3.1.3) : 1, intérieur chauffé ; 2, abrité non chauffé ; 3, exposé aux intempéries. "
                "Elle fixe k_mod (la résistance) et k_def (le fluage).", full=True)
    s += C.aide(f"{C.esc(co['libelle_service'])} · k<sub>mod</sub> {C.nb(co['kmod_permanent'])} (G) et {C.nb(co['kmod_variable'])} (G + Q) · "
                f"k<sub>def</sub> = {C.nb(co['kdef'])}")
    s += C.ligne2(("γ<sub>G</sub>", C.num(["gamma_G"], p["gamma_G"], ""), "Le coefficient partiel des charges permanentes (NBN EN 1990 ANB, 6.10 : 1,35)."),
                  ("γ<sub>Q</sub>", C.num(["gamma_Q"], p["gamma_Q"], ""), "Le coefficient partiel de la charge variable (NBN EN 1990 ANB, 6.10 : 1,50)."))
    verrou = "" if libre else f" Fixée par la nature de Q ({D.LIBELLES_NATURE[ch['nature']]}) : choisir « Autre » pour la saisir."
    s += C.ligne("Durée Q", C.choix(["charges", "duree"], ch["duree"], [(k, D.LIBELLES_DUREE[k].capitalize()) for k in D.DUREES], desactive=not libre), "",
                 "La classe de durée de la charge variable (EN 1995-1-1 tableau 2.1) : elle donne le k_mod de la combinaison G + Q." + verrou,
                 full=True)
    verrou = "" if libre else f" Fixé par la nature de Q ({D.LIBELLES_NATURE[ch['nature']]}) : choisir « Autre » pour le saisir."
    s += C.ligne("de Q", C.num(["charges", "psi2"], ch["psi2"], "", desactive=not libre), "ψ<sub>2</sub>",
                 "La part quasi permanente de Q : elle flue avec G dans la flèche finale, w_fin = w_g·(1 + k_def) + w_q·(1 + ψ₂·k_def)." + verrou)
    s += C.ligne("Effet système", C.case(["systeme"], p["systeme"], f"k<sub>sys</sub> = {C.nb(D.K_SYS)}"), "",
                 "EN 1995-1-1 §6.6 : f_m,d et f_v,d majorés de 10 % pour des pièces semblables, régulièrement espacées (60 cm environ) "
                 "et liées par un platelage continu capable de répartir les charges. À ne cocher que si ces conditions sont réunies.")
    s += C.ligne2(("w<sub>inst</sub> ≤", C.choix(["lim_inst"], p["lim_inst"], [(n, f"L/{n}") for n in D.LIMITES_INST]),
                   "La limite de la flèche instantanée sous G + Q (EN 1995-1-1 tableau 7.2 : L/300 à L/500 sur deux appuis)."),
                  ("w<sub>net,fin</sub> ≤", C.choix(["lim_fin"], p["lim_fin"], [(n, f"L/{n}") for n in D.LIMITES_FIN]),
                   "La limite de la flèche finale nette, fluage compris, contre-flèche déduite (tableau 7.2 : L/250 à L/350)."))
    s += C.ligne("contre-flèche", C.num(["contre_fleche"], p["contre_fleche"], "mm"), "w<sub>c</sub>",
                 "La contre-flèche donnée à la poutre à la fabrication (lamellé-collé) : w_net,fin = w_fin − w_c. 0 : aucune.")
    return s


def formulaire(etat: Dict, r: Optional[Dict] = None) -> str:
    """Les quatre rubriques de la poutre ouverte : 1 Poutre, 2 Charges, 3 Appuis et maintien, 4 Hypothèses (secondaire).
    ``r`` : le calcul de la poutre ouverte (``MO.verifier``), refait s'il manque."""
    p = _poutre(etat)
    r = r or MO.verifier(p)
    geo, rc, ap = r["geometrie"], r["charges"], p["appui"]
    r1 = f"{C.esc(p['classe'])} · {_section_txt(geo)} · L {C.nb(p['L'])} m"
    r2 = f"G {C.nb(round(rc['G'], 3))} · Q {C.nb(round(rc['Q'], 3))} kN/m"
    r3 = f"a {C.nb(ap['a'])} mm · {dict(SEGMENTS_MAINTIEN)[p['maintien']['type']].lower()}"
    r4 = f"service {p['service']} · L/{p['lim_inst']} · L/{p['lim_fin']}"
    return (C.section(1, "Poutre", r1, _rubrique_poutre(p, r), True, "po")
            + C.section(2, "Charges", r2, _rubrique_charges(p, r), True, "ch")
            + C.section(3, "Appuis et maintien", r3, _rubrique_appuis(p, r), False, "ap")
            + C.section(4, "Hypothèses", r4, _rubrique_hypotheses(p, r), False, "hy", secondaire=True))


def _dims(r: Dict) -> str:
    """« · 75 × 225 · C24 » : les dimensions d'une poutre dans la liste de la note (HTML)."""
    if "geometrie" not in r:
        return ""
    return f"· {_section_txt(r['geometrie'])} · {C.esc(r['materiau']['classe'])}"


def arbre(etat: Dict, r: Optional[Dict] = None, resultats: Optional[List[Dict]] = None) -> str:
    """La colonne de gauche COMPACTE (api_web/etabli_compact.arbre_compact) : la ligne de la poutre ouverte (son nom,
    qu'un clic modifie ; la liste des poutres « nom · section · classe » ; copier, supprimer), puis ses rubriques."""
    resultats = resultats or [_calculer(x) for x in etat["poutres"]]
    elements = [(p.get("nom") or f"P{i + 1}", _dims(x), False) for i, (p, x) in enumerate(zip(etat["poutres"], resultats))]
    return E.arbre(E.POUTRE, elements, etat["actif"], formulaire(etat, r), "poutre", compact=True)


# ------------------------------------------------------------------ le résultat
def _calculer(p: Dict) -> Dict:
    try:
        return MO.verifier(p)
    except ValueError as e:                                     # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(e)}


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE (l'accueil, la liste des notes) : sa poutre la plus chargée."""
    vals = [(x["taux_max"], i) for i, x in enumerate(resultats) if "taux_max" in x]
    if not vals:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    t, i = max(vals)
    x = resultats[i]
    niv = _niv_res(x)
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1],
            "gouvernant": f"{etat['poutres'][i].get('nom') or 'Poutre'} · {COURTS[x['gouvernant']['cle']].lower()}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """Rien : les poutres de la note sont la LIGNE de l'élément et les rubriques de la colonne de gauche (``arbre``)."""
    return ""


def _comparaison(r: Dict, v: Dict, long: bool = False) -> str:
    """Ce qu'une vérification compare, en clair : « σ_m,d = 6,65 ≤ 14,77 N/mm² » ; ``long`` : la limite avec son symbole
    (le verdict, où la place ne manque pas). À l'ELS, la flèche est aussi dite en « L/n »."""
    u = v["unite"]
    signe = "≤" if v["etat"] == MO.OK else "&gt;"
    if v["groupe"] == MO.ELS:
        fl = r["fleches"]
        n, rapport = (fl["n_inst"], fl["rapport_inst"]) if v["cle"] == "fleche_inst" else (fl["n_fin"], fl["rapport_fin"])
        dite = f" (L/{C.fr(rapport, 0)})" if rapport else ""
        return f"{_symb(v['symbole'])} = {_n(v['valeur'], u)}{dite} {signe} L/{n} = {_nu(v['limite'], u)}"
    lim = f"{_symb(v['symbole_limite'])} = " if long else ""
    return f"{_symb(v['symbole'])} = {_n(v['valeur'], u)} {signe} {lim}{_nu(v['limite'], u)}"


def _piece(r: Dict, v: Dict) -> str:
    """Ce que la 1ʳᵉ ligne d'une vérification rappelle à côté de son nom (HTML court)."""
    cle = v["cle"]
    if cle == "flexion":
        return _section_txt(r["geometrie"])
    if cle == "cisaillement":
        c = _comb(r, v["combinaison"])
        return f"V<sub>d</sub> = {_nu(c['V_d'], 'kN')}"
    if cle == "appui":
        return f"a = {C.nb(r['appui']['a'])}{C.NBSP}mm"
    return f"L/{r['fleches']['n_inst' if cle == 'fleche_inst' else 'n_fin']}"


def _verdict(r: Dict) -> str:
    v = next(x for x in r["verifications"] if x["cle"] == r["gouvernant"]["cle"])
    return (f'<div class="verdict" data-calcul="{v["cle"]}" title="Voir le calcul">{C.chip(C.pct(r["taux_max"]), _niv_res(r), xl=True)}<div class="gov">'
            f'<b>{C.esc(COURTS[v["cle"]])}</b>{_comparaison(r, v, long=True)}<br>{_comb_txt(r, v)}</div></div>')


def _etat_groupe(vs: List[Dict]) -> Tuple[str, str]:
    """(couleur, état) de l'en-tête d'un bloc : rouge dès qu'une de ses vérifications ne tient pas."""
    t = max(v["taux"] for v in vs)
    if any(v["etat"] == MO.NOK for v in vs):
        return "nok", f"Non conforme · {_pct(t)}"
    return "ok", f"Conforme · {_pct(t)}"


def _blocs(r: Dict, leger: str) -> str:
    """Les deux blocs thématiques : « ELU · résistance » (les combinaisons, puis flexion, cisaillement, appui) et
    « ELS · flèches » (instantanée, finale nette) ; chaque ligne dit sa comparaison et la combinaison qui la gouverne."""
    out = ""
    gov = r["gouvernant"]["cle"]
    co, fl = r["coefficients"], r["fleches"]
    for i, (groupe, titre) in enumerate(GROUPES):
        vs = [v for v in r["verifications"] if v["groupe"] == groupe]
        rows = ""
        for v in vs:
            suite = f" · {_comb_txt(r, v, court=True)}" if groupe == MO.ELU else ""
            if v["cle"] == "flexion" and r["deversement"]["kcrit"] < 1:
                suite = f" · k<sub>crit</sub>{C.NBSP}{_n(r['deversement']['kcrit'])}" + suite
            rows += C.ligne_verif(COURTS[v["cle"]], _comparaison(r, v) + suite, C.pct(v["taux"]), _niv(v), calcul=v["cle"],
                                  pk="section" if v["cle"] == "flexion" else "", piece=_piece(r, v), gov=v["cle"] == gov,
                                  leger=leger if v["cle"] == "flexion" else "")
        if groupe == MO.ELU:
            if not r["appui"]["verifie"]:
                rows += C.ligne_verif(COURTS["appui"], "compression à l'appui retirée de la note", None, "c", etat_txt="non vérifié")
            # les combinaisons, alignées en colonnes (css/modules/bois.css) : le nom, k_mod, M_d, V_d
            efforts = '<div class="bo-combs">' + "".join(
                f'<span class="n" title="{C.esc(c["libelle_duree"])}">{C.esc(c["nom"])}</span><span>k<sub>mod</sub>{C.NBSP}{C.nb(c["kmod"])}</span>'
                f'<span>M<sub>d</sub>{C.NBSP}{_nu(c["M_d"], "kN·m")}</span><span>V<sub>d</sub>{C.NBSP}{_nu(c["V_d"], "kN")}</span>'
                for c in r["combinaisons"]) + "</div>"
        else:
            efforts = (f"combinaison caractéristique · k<sub>def</sub> = {C.nb(co['kdef'])} · ψ<sub>2</sub> = {C.nb(co['psi2'])}"
                       + (f" · contre-flèche {_nu(fl['w_c'], 'mm')}" if fl["w_c"] else ""))
        cls, etat_txt = _etat_groupe(vs)
        out += E.bloc(i, titre, cls, etat_txt, efforts, rows, theme=True)
    return out


def _message(a: Dict) -> str:
    """Un avertissement du moteur : son début (jusqu'aux deux-points) en gras, les symboles avec leurs indices."""
    tete, sep, suite = a["message"].partition(" : ")
    return f"<b>{_symb(tete)}</b>{_symb(sep + suite)}" if sep else _symb(a["message"])


def _avis(cle: Optional[str], r: Dict) -> List[str]:
    """Le texte de l'avis d'ouverture (TEXTE simple : l'écran l'échappe), ou rien."""
    if cle in ("v1", "v1_nom"):
        return [AVIS_V1 + (AVIS_NOM if cle == "v1_nom" else "")]
    if cle == "section":
        geo, mat = r["geometrie"], r["materiau"]
        return [f"Le bois change de type : la section {geo['designation']} mm n'est pas une section standard du {D.COURTS_TYPE[mat['type']]} ; "
                "elle est gardée « sur mesure ». Les sections standard de ce bois sont dans le sélecteur (ligne « Flexion » du résultat)."]
    return []


def _hypotheses_html(r: Dict) -> str:
    """Sous les blocs : les hypothèses du moteur, puis ce que le module ne couvre pas — des lignes discrètes."""
    lignes = "".join(C.aide(_symb(h)) for h in r["hypotheses"])
    non = C.aide("<b>Non couvert par ce module</b> : " + _symb(" ; ".join(r["non_couvert"])) + ".")
    return f'<div class="et-hyps">{lignes}{non}</div>'


# ------------------------------------------------------------------ les fenêtres de calcul
def _fx(f: Dict) -> Dict:
    """Une formule du moteur → « symbole = expression = valeurs substituées = résultat » : son gabarit, où chaque terme
    est remplacé par sa valeur mise en forme ; une valeur LUE (sans gabarit) dit seulement d'où elle vient."""
    termes = {t["symbole"]: t["valeur"] for t in f["termes"]}
    parts = [f["symbole"], f["expression"]]
    if f["gabarit"] is not None:
        parts.append(_ACCOLADES.sub(lambda m: _v(termes[m.group(1)]), f["gabarit"]))
    if f["cle"] == "taux":
        return {"parts": parts, "res": _pct(f["resultat"]), "niveau": C.niveau(f["resultat"])}
    return {"parts": parts, "res": _nu(f["resultat"], f["unite"]), "niveau": ""}


def _groupe(r: Dict, bloc: Dict, gouverne: Optional[str] = None) -> str:
    """Un groupe du détail d'une fenêtre : le titre du bloc du moteur, sa combinaison (« G + Q · moyen terme », et
    « gouverne » quand c'est elle), ses lignes."""
    sous = ""
    c = _comb(r, bloc["combinaison"])
    if c is not None:
        propre = bloc["cle"].startswith("combinaison_")
        sous = C.esc(c["libelle_duree"] if propre else f"{c['nom']} · {c['libelle_duree']}") + (" · gouverne" if gouverne == c["cle"] else "")
    st = f'<span class="ck-comb">{sous}</span>' if sous else ""
    return f'<div class="ck-g"><div class="ck-t">{C.esc(bloc["titre"])}{st}</div>{C.lignes_fx([_fx(f) for f in bloc["formules"]])}</div>'


def _hyps(r: Dict) -> List[str]:
    """Les hypothèses d'une fenêtre, en pastilles : le bois, la classe de service, k_mod des combinaisons, k_def, γ_M,
    la nature de Q."""
    mat, co = r["materiau"], r["coefficients"]
    kmod = " · ".join(f"{C.nb(c['kmod'])} ({C.esc(c['nom'])}, {C.esc(c['libelle_duree'])})" for c in r["combinaisons"])
    out = [f"Bois <b>{C.esc(mat['classe'])}</b> · {C.esc(mat['libelle_type'])} · {C.esc(mat['norme'])}",
           f"Classe de service <b>{co['service']}</b> · {C.esc(co['libelle_service'])}",
           f"k<sub>mod</sub> = {kmod}",
           f"k<sub>def</sub> = {C.nb(co['kdef'])}",
           f"γ<sub>M</sub> = {C.nb(co['gamma_M'])} · γ<sub>G</sub> = {C.nb(co['gamma_G'])} · γ<sub>Q</sub> = {C.nb(co['gamma_Q'])}",
           f"Q : {C.esc(co['libelle_nature'])} · ψ<sub>2</sub> = {C.nb(co['psi2'])}"]
    if co["systeme"]:
        out.append(f"Effet système k<sub>sys</sub> = {C.nb(co['ksys'])}")
    return out


def _lignes_tableau(r: Dict, v: Dict, marquer) -> List[Dict]:
    """Les lignes du tableau d'une vérification : une par combinaison à l'ELU (celle que ``marquer`` désigne est mise
    en avant), une seule à l'ELS."""
    u, f = v["unite"], _symb(v["enonce"])
    if v["groupe"] == MO.ELS:
        fl = r["fleches"]
        n = fl["n_inst"] if v["cle"] == "fleche_inst" else fl["n_fin"]
        return [{"nom": C.esc(COURTS[v["cle"]]), "sous": "ELS caractéristique" + ("" if v["cle"] == "fleche_inst" else " · fluage compris"),
                 "formule": f, "valeur": f"{_symb(v['symbole'])} = {_nu(v['valeur'], u)}", "limite": f"L/{n} = {_nu(v['limite'], u)}",
                 "taux": C.pct(v["taux"]), "niveau": _niv(v), "gov": marquer(v, None)}]
    return [{"nom": C.esc(COURTS[v["cle"]]), "sous": f"{C.esc(d['nom'])} · {C.esc(D.LIBELLES_DUREE[d['duree']])} · k<sub>mod</sub> = {C.nb(d['kmod'])}",
             "formule": f, "valeur": f"{_symb(v['symbole'])} = {_nu(d['valeur'], u)}",
             "limite": f"{_symb(v['symbole_limite'])} = {_nu(d['limite'], u)}", "taux": C.pct(d["taux"]), "niveau": _niv(d),
             "gov": marquer(v, d["combinaison"])} for d in v["detail"]]


def _pied(r: Dict) -> str:
    """Le pied d'une fenêtre : les hypothèses du moteur, ce qui n'est pas couvert."""
    hyp = "".join(f"<li>{_symb(h)}</li>" for h in r["hypotheses"])
    return (f'<div class="ck-autres bo-pied"><b>Hypothèses</b><ul>{hyp}</ul>'
            f'<b>Non couvert par ce module</b> : {_symb(" ; ".join(r["non_couvert"]))}.</div>')


def _blocs_de(r: Dict, v: Dict) -> List[str]:
    """Les blocs du détail d'une vérification, POUR CHAQUE combinaison (la gouvernante d'abord) : ceux que le moteur
    donne pour la combinaison qui gouverne, et les mêmes pour l'autre."""
    if v["combinaison"] is None:
        return list(v["blocs"])
    autres = [c["cle"] for c in r["combinaisons"] if c["cle"] != v["combinaison"]]
    out = []
    for k in v["blocs"]:
        out.append(k)
        if k.endswith("_" + v["combinaison"]):
            out += [k[:-len(v["combinaison"])] + a for a in autres]
    return out


def _volets(r: Dict, nom: str) -> str:
    """Les fenêtres de calcul : une par vérification (son tableau, puis le détail qui y mène, combinaison par
    combinaison), et « tout le calcul » de chaque bloc (s0 : ELU ; s1 : ELS)."""
    geo, mat = r["geometrie"], r["materiau"]
    hyps = _hyps(r)
    par_cle = {b["cle"]: b for b in r["calcul"]}
    base = f"{_section_txt(geo)} mm · {C.esc(mat['classe'])} · L = {C.qte(geo['L'], 'm', 2)}"
    pied = _pied(r)
    out = ""
    for v in r["verifications"]:
        lignes = _lignes_tableau(r, v, lambda vv, c: c == vv["combinaison"])
        detail = "".join(_groupe(r, par_cle[k], v["combinaison"]) for k in _blocs_de(r, v))
        out += C.fenetre_calcul(v["cle"], f"{nom} · {C.esc(COURTS[v['cle']].lower())}", f"{base} · EN 1995-1-1 {C.esc(v['article'])}", hyps, lignes,
                                detail, pied)
    for i, (groupe, titre) in enumerate(GROUPES):
        vs = [v for v in r["verifications"] if v["groupe"] == groupe]
        t_max = max(v["taux"] for v in vs)
        lignes = [li for v in vs for li in _lignes_tableau(r, v, lambda vv, c, t=t_max: c == vv["combinaison"] and vv["taux"] == t)]
        cles = []
        for v in vs:
            cles += [k for k in _blocs_de(r, v) if k not in cles]
        ordre = [b["cle"] for b in r["calcul"] if b["cle"] in cles]          # l'ordre de la note : celui du moteur
        gouv = {v["cle"]: v["combinaison"] for v in vs}
        detail = "".join(_groupe(r, par_cle[k], gouv.get(k.rsplit("_", 1)[0])) for k in ordre)
        out += C.fenetre_calcul(f"s{i}", f"{nom} · {C.esc(titre)}", f"{base} · tout le calcul", hyps, lignes, detail, pied)
    return out


# ------------------------------------------------------------------ les vues
def _entete(r: Dict, titre: str, legende: str) -> str:
    """L'en-tête technique d'une vue : la section et le bois ; les charges ; la combinaison qui gouverne l'ELU, l'appui
    et le maintien ; à droite la bande des efforts de calcul (M_d, V_d, la réaction d'appui R_d)."""
    geo, mat, ch, co, ef, ap = r["geometrie"], r["materiau"], r["charges"], r["coefficients"], r["efforts"], r["appui"]
    c = _comb(r, ef["combinaison"])
    bande = [("M", "d", ef["M_d"], "kN·m"), ("V", "d", ef["V_d"], "kN"), ("R", "d", ef["R"], "kN")]
    pp = f" (dont poids propre {C.fr(ch['poids_propre'], 2)})" if ch["poids_propre_compte"] else " (poids propre non compté)"
    lignes = [f"g = {C.fr(ch['g'], 2)} kN/m{pp} · q = {C.fr(ch['q'], 2)} kN/m · {co['libelle_nature']} · classe de service {co['service']}",
              f"ELU · combinaison {c['nom']} ({c['libelle_duree']}) · appui a = {C.nb(ap['a'])} mm · {r['deversement']['libelle']}"]
    sous = f"{_section_txt(geo)} mm · {C.esc(mat['classe'])} · L = {C.qte(geo['L'], 'm', 2)}"
    return E.entete(titre, sous, lignes, bande, legende)


def _vues(r: Dict, ident: str) -> List[Dict]:
    """Les trois vues : le schéma et les efforts, la déformée, la section — chacune sous son en-tête technique."""
    fl = r["fleches"]
    vues = [("schema", "Schéma et efforts", DB.poutre, False,
             "charges g (permanente, poids propre compris) et q (variable) · appuis à leur longueur · V et M de la combinaison qui gouverne"),
            ("deformee", "Déformée", DB.deformee, False,
             f"w inst. {_n(fl['w_inst'], 'mm')} mm · w net,fin {_n(fl['w_net_fin'], 'mm')} mm · limites en tirets · flexion et effort tranchant"),
            ("section", "Section", DB.section, True,
             "section à l'échelle, cotes en mm · fil du bois · maintien latéral · contraintes de calcul et leurs limites")]
    return [{"cle": cle, "titre": titre, "court": titre, "svg": fn(f"{ident}_{cle}", r), "entete": _entete(r, titre.upper(), legende),
             "legende": legende, "echelle": echelle} for cle, titre, fn, echelle, legende in vues]


def resultat(etat: Dict) -> Dict:
    """Le résultat de la poutre ouverte : la tête (le nom, « 75 × 225 · C24 · L 4,00 m »), l'avis d'ouverture (dit une
    fois, retiré de l'état ; dans « avis » aussi, pour le téléphone), le verdict, les deux blocs, les avertissements,
    les hypothèses et ce qui n'est pas couvert, la masse ; les fenêtres de calcul, les vues, la colonne de gauche, le
    résumé de la note."""
    cle_avis = etat.pop(CLE_AVIS, None)
    p = _poutre(etat)
    r = MO.verifier(p)
    avis = _avis(cle_avis, r)
    resultats = [r if i == etat["actif"] else _calculer(x) for i, x in enumerate(etat["poutres"])]
    geo, mat, ch = r["geometrie"], r["materiau"], r["charges"]
    nom = C.esc(p.get("nom") or "Poutre")
    tete = E.tete_resultats(nom, f"{_section_txt(geo)} · {C.esc(mat['classe'])} · L {C.qte(geo['L'], 'm', 2)}")
    # la plus légère qui tient (l'étoile de la ligne « Flexion », qui ouvre le sélecteur)
    opt = MO.optimal(p)
    leger = (f"La plus légère qui tient : {opt['designation']} ({C.pc(C.pct(opt['taux']))} %)"
             if opt and not (p["standard"] and opt["nom"] == p["section"]) and (opt["A"] < geo["A"] or r["statut"] == MO.NOK) else "")
    messages = "".join(C.message(_message(a)) for a in r["avertissements"])
    pp = f"{C.qte(ch['poids_propre'], 'kN/m', 3)}{'' if ch['poids_propre_compte'] else ' (non compté)'}"
    pied = (f'<div class="pfoot"><div class="mass"><span>Masse · {C.qte(geo["longueur"] / 1000, "m", 2)} de pièce</span>'
            f'<b>{C.qte(r["masse"], "kg", 1)} · {C.qte(r["masse_lineique"], "kg/m", 2)}</b></div>'
            f'<div class="mass"><span>Poids propre</span><b>{pp}</b></div><div class="legend">{C.legende()}</div></div>')
    nivs = [_niv_res(x) if "taux_max" in x else "x" for x in resultats]
    statut = C.statut(next((n for n in ("x", "r") if n in nivs), max(nivs, key=lambda n: {"p": 0, "f": 1}.get(n, 0)))) if len(resultats) > 1 else ""
    corps = (f'<div class="pbody">{tete}{"".join(C.message(C.esc(a)) for a in avis)}{_verdict(r)}<div class="et-blocs">{_blocs(r, leger)}</div>'
             f'{messages}{_hypotheses_html(r)}</div>{pied}')
    return {"resultat": corps, **({"avis": avis} if avis else {}), "volets": _volets(r, nom), "vues": _vues(r, f"bo{etat['actif']}"),
            "phrase": "", "statut_html": statut, "resume": resume_note(etat, resultats), "donnees": arbre(etat, r, resultats),
            "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur de sections
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les sections standard du TYPE du bois de la poutre ouverte, chacune recalculée par le moteur avec ses charges,
    son appui et son maintien (``moteur.candidats_sections``), TRIÉES PAR MASSE : ✓ la retenue, ★ la plus légère qui
    tient ; rouge quand elle ne tient pas ; sous la liste, ses caractéristiques et la vérification qui la gouverne."""
    p = _poutre(etat)
    cands = MO.candidats_sections(p)
    typ = EN.type_classe(p["classe"])
    i_o = next((k for k, c in enumerate(cands) if c["optimal"]), 0)
    i_c = next((k for k, c in enumerate(cands) if c["retenue"]), i_o)

    def prop(c):
        return (f"{C.esc(c['designation'])} mm · A = {C.qte(c['A'] / 100, 'cm²', 1)} · W = {C.qte(c['W'] / 1e3, 'cm³', 1)} · "
                f"I = {C.qte(c['I'] / 1e4, 'cm⁴', 0)} · gouverne : {C.esc(COURTS[c['gouvernant']].lower())} ({_pct(c['taux'])})"
                + ("" if c["ok"] else " · <b>ne tient pas</b>"))
    rows = [{"nom": c["designation"], "col": C.fr(c["masse"], 2), "taux": C.pct(c["taux"]), "niveau": "r" if not c["ok"] else C.niveau(c["taux"]),
             "cur": c["retenue"], "opt": c["optimal"], "prop": prop(c)} for c in cands]
    sous = (f"{len(cands)} sections · par masse · calculées avec vos charges"
            + ("" if any(c["ok"] for c in cands) else " · aucune ne tient"))
    return C.selecteur(role, f"Section · {C.esc(D.COURTS_TYPE[typ])} {C.esc(p['classe'])}", sous, "", rows, max(0, min(i_o, i_c) - 4), "kg/m",
                       etoile="la plus légère qui tient")

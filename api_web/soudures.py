# -*- coding: utf-8 -*-
"""Le module « Soudures » du site : l'état d'une note (des CAS, celui du moteur ``acier/soudures/etat``), les champs de
saisie, le résultat cordon par cordon, les fenêtres de calcul (les formules de la méthode choisie, substituées), les
quatre lectures d'une même soudure (Eurocode directionnelle et simplifiée, méthode ELS du bureau, k Eurocode), les
sélecteurs (la gorge, le profilé) et les dessins de dessinateur — rendus en fragments HTML.

Le calcul est celui du moteur ``acier/soudures`` (sans Streamlit), le MÊME que l'écran Streamlit modules/soudures.py
appelle : ``etat.verifier(cas)``. Ce fichier ne calcule rien ; il reprend seulement les quelques effets de saisie que
l'écran Streamlit porte dans son code d'écran, à l'identique (tests/test_api_web_soudures.py les rejoue contre lui, avec un
faux « st » qui exécute l'écran réel) :

- la case « par tronçons » : cochée, n = max(1, n) ; décochée, n = 0 (``_troncons``) ;
- le tableau des cordons libres (C8) relu ligne par ligne (``elements_depuis_lignes``) : noms complétés, valeurs
  manquantes prises aux défauts du type, côté ±1, retours d'angle remis à « non » ;
- « Prendre la valeur Eurocode » : k = round(k équivalent, 1) ; « Revenir à 100 » ; « ＋ Cas » : une console C3 qui
  garde la méthode du cas ouvert ; le changement de configuration (``etat.changer_config``) ;
- les minimums des champs (une valeur sous le minimum, que Streamlit n'accepte pas, est REFUSÉE avec un message) ;
- une note venue d'un fichier ne porte que ce que l'écran laisse saisir (``controler_cas``) : là où l'écran Streamlit
  afficherait autre chose que ce qu'il calcule (valeur hors d'un menu, côté ±2 ramené à +1…), le site refuse et le dit.

La méthode, la preuve de parité, ce qui est couvert et les défauts trouvés dans l'écran Streamlit :
docs/acier/SOUDURES_SITE.md.

Les copies de l'écran Streamlit (textes d'aide, minimums, libellés) sont NOMMÉES ici et un test les lie à l'écran.

Des retouches d'ÉCRITURE seulement (jamais de calcul), l'écran Streamlit et la note gardent celle du moteur (défauts
écrits dans docs/acier/SOUDURES_SITE.md) :

- un nombre négatif élevé au carré s'imprime « −157,9² » (qui se lit −(157,9²)) ; le site écrit « (−157,9)² »
  (``_parentheses``) ;
- les symboles des dessins et des messages prennent leur indice (« h_p » → h avec p en indice : ``indices_svg``,
  ``_texte_indices``) ; le signe moins est « − » (``_fr``) ;
- dans la fenêtre de calcul et le verdict, valeurs et limites arrondies comme la formule voisine du moteur
  (``_fr_moteur`` : F_adm = 490,5 s'écrit « 491 » partout) ;
- sous les efforts réduits, la correspondance entre les efforts saisis et le repère du joint, LUE au moteur
  (``correspondance_axes`` : en C5, V_z saisi est le V_y des dessins et des formules).

L'écran est l'établi COMPACT (api_web/etabli_compact.py, la mise en forme des poutres — 12/11/2026) : à gauche la ligne
du cas ouvert (son nom, la liste, copier, supprimer) et ses rubriques en accordéon — Assemblage, Géométrie, Soudure,
Efforts, Aciers, puis deux rubriques secondaires (Hypothèses ; Note PDF et reprise) ; au centre les vues sous leur
en-tête technique (la configuration, la méthode ; les efforts saisis dans la bande) ; à droite la tête, le verdict et
deux blocs : « Cordons » (une ligne par cordon : ce qu'il compare à son point le plus chargé, sa gorge qui se choisit)
et « Les quatre lectures ». Un cas neuf s'appelle court (« S1 ») : le nom du moteur, « C3 · Console fléchie », ne tient
pas dans la ligne de l'élément d'une tablette — la configuration se lit dans la liste et en tête des résultats.
"""
import copy
import html
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from acier.assemblages.poutre_poutre.doubles_cornieres import formules as KF
from acier import formats as AF
from acier.profiles import catalogue as CAT
from acier.profiles import familles as FA
from acier.profiles import geometrie as _GEO  # noqa: F401 — chargé à la demande par le moteur (pourtour d'un tube) : dans le paquet du site
from acier.soudures import configurations as CF
from acier.soudures import criteres as K
from acier.soudures import dessin as DS
from acier.soudures import etat as ET
from acier.soudures import formules as FO
from acier.soudures import methode as MT
from acier.soudures import schemas as SC

from . import commun as C
from . import etabli_compact as E
from . import note_cas as NC

MODULE = "soudures"
PAGE = "soudures"
GENRE = E.Genre("Cas", "le cas", "ce cas", "un cas")
NOM_DEFAUT = "S1"                # le nom d'un cas neuf : court (il tient dans la ligne de l'élément, tablette et téléphone)
CLASSES_BLOC = {"p": "ok", "f": "warn", "r": "nok"}

# ------------------------------------------------------------------ les copies NOMMÉES de l'écran Streamlit
# (tests/test_api_web_soudures.py, section A, les relit dans le source de modules/soudures.py)
MODES = {K.EUROCODE: "Eurocode (ELU)", K.ELS: "Méthode ELS (bureau)"}
CRITERES = {K.DIRECTIONNELLE: "directionnelle (§4.5.3.2)", K.SIMPLIFIEE: "simplifiée (§4.5.3.3)"}
COTES_C4 = ("gauche", "droite", "bas", "haut")
FRONTAUX = {0: "aucun", 1: "un (bout du plat B)", 2: "deux (bouts de B et de A)"}
SEMELLES = {"deux faces": "deux faces (ext. + int.)", "extérieur": "face extérieure seule"}
PENETRATIONS = {"pleine": "pleine (la pièce gouverne)", "partielle": "partielle (gorge = pénétration p)"}
G_OPTIONS = {9.81: "9,81 N/kg", 10.0: "10 N/kg"}
JONCTION_AIDE = ("Cordons d'angle : gorge a sur chaque face. Pénétration partielle (EN 1993-1-8 §4.7.2) : pénétration p par face ; "
                 "avec un cordon de renfort a_r, cordon à pénétration profonde de gorge p + a_r (§4.5.2(3)) ; deux faces avec 2p ≥ t et "
                 "t − 2p ≤ min(t/5 ; 3 mm) : traitée comme pleine (§4.7.3). Pleine pénétration (§4.7.1) : la soudure vaut la pièce, "
                 "sa section t × l est vérifiée au métal de base (fy/γM0) — le cas d'une cornière reconstituée de deux plats épais.")
TRONCONS_AIDE = ("Au lieu d'un cordon continu : n tronçons de l mm, aux deux bouts et régulièrement entre (pourtour : "
                 "régulièrement, le premier en haut). EN 1993-1-8 §4.3.2.2 : espacement ≤ min(200 mm ; 12 t en compression ; 16 t en traction).")
C8_AIDE = ("Chaque ligne est un **segment** (P1 → P2) ou un **arc** (centre, rayon, angles en degrés, sens trigonométrique ; "
           "0 → 360 = cercle). Côté : la pièce rapportée est à gauche (+1) ou à droite (−1) du sens de parcours — "
           "pour un cercle, +1 = la pièce est à l'intérieur. Tronçons n = 0 : élément continu.")
# les minimums des champs numériques de l'écran Streamlit (min_value), par chemin ; absent : pas de minimum
MINIMUMS: Dict[Tuple, float] = {
    ("params", "b"): 10.0, ("params", "tB"): 1.0, ("params", "L_rec"): 10.0, ("params", "tA"): 1.0, ("params", "a"): 1.0,
    ("params", "l"): 10.0, ("params", "h"): 10.0, ("params", "t"): 1.0, ("params", "e"): 0.0, ("params", "bx"): 10.0,
    ("params", "by"): 10.0, ("params", "tp"): 1.0, ("params", "p"): 1.0, ("params", "renfort"): 0.0,
    ("params", "a_f"): 1.0, ("params", "p_f"): 1.0, ("params", "renfort_f"): 0.0,
    ("params", "a_w"): 1.0, ("params", "p_w"): 1.0, ("params", "renfort_w"): 0.0,
    ("methode", "gamma_M2"): 1.0, ("methode", "k"): 1.0, ("methode", "gamma_F"): 1.0,
    ("troncons", "n"): 1, ("troncons", "l"): 1.0,
    ("element", "a"): 1.0, ("element", "n"): 0, ("element", "l"): 1.0,
}
NUMERIQUES = {"C1": ("b", "tB", "L_rec", "tA", "a"), "C2": ("l", "tB", "tA", "a", "p", "renfort"),
              "C3": ("h", "t", "e", "a", "p", "renfort"), "C4": ("bx", "by", "t", "a", "xV", "yV"),
              "C5": ("tp", "a_f", "p_f", "renfort_f", "a_w", "p_w", "renfort_w"), "C6": ("tp", "a", "p", "renfort"),
              "C7": ("b", "t", "p"), "C8": ("xR", "yR", "zR")}
BOOLEENS = {"C1": ("retours", "lateraux"), "C2": ("deux_cordons", "retours"), "C3": ("deux_cordons", "retours"), "C4": ("retours",),
            "C5": ("ame", "retours"), "C6": (), "C7": ("deux_faces",), "C8": ()}
TRONCONS = {"C1": ("troncons",), "C2": ("troncons",), "C3": ("troncons",), "C4": ("troncons",),
            "C5": ("troncons_semelles", "troncons_ame"), "C6": ("troncons",), "C7": (), "C8": ()}
COORDS_SEGMENT = ("x1", "y1", "x2", "y2")
COORDS_ARC = ("xc", "yc", "r", "ang1", "ang2")
# la portée d'un nombre saisi (mm, kN, kN·m, coefficients) : au-delà, le moteur déborde (1e300 mm → NaN, puis une
# erreur Python au lieu d'un résultat) ; l'écran Streamlit l'accepte et s'arrête sur une trace. Le site refuse et le dit.
BORNE = 1e6
# le nombre de tronçons d'un cordon : au-delà, une ligne de résultat et une fenêtre de calcul PAR tronçon rendent l'écran
# illisible et le navigateur lent (200 tronçons : 400 lignes, 2,5 Mo de fenêtres) — écart voulu, l'écran Streamlit n'a
# pas de maximum
N_TRONCONS_MAX = 100
# les gorges proposées par le sélecteur (mm) ; la valeur saisie s'y ajoute toujours
GORGES = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0)
PENETRATIONS_P = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0)
# les libellés courts des efforts (le texte complet, le sens, est l'aide du moteur : CF.EFFORTS)
EFFORTS_COURTS = {      # courts : ils tiennent à côté du champ sur une tablette (le sens complet est l'aide du moteur)
    "C1": {"N": "traction", "V": "transversal"},
    "C2": {"N": "arrachement", "Va": "le long", "Vp": "en travers", "M": "flexion"},
    "C3": {"V": "verticale", "N": "horizontale"},
    "C4": {"V": "verticale", "H": "horizontale", "N": "arrachement"},
    "C5": {"N": "normal", "Vz": "selon l'âme", "Vy": "selon les ailes", "My": "axe fort", "Mz": "axe faible", "T": "torsion"},
    "C6": {"N": "normal", "Vz": "vertical", "Vy": "horizontal", "My": "axe horizontal", "Mz": "axe vertical", "T": "torsion"},
    "C7": {"N": "traction", "V": "cisaillement", "M": "flexion"},
    "C8": {"N": "hors plan", "Vx": "selon x", "Vy": "selon y", "Mx": "flexion x", "My": "flexion y", "Mz": "torsion"},
}
VUES_COURTES = {"Vue en plan": "Plan", "Coupe A-A": "Coupe A-A", "Élévation": "Élévation", "Coupe horizontale": "Coupe",
                "Vue de face": "Face", "Vue du plan de joint": "Plan de joint"}
GROUPES_PROFIL = {"C5": "I/H", "C6": "creux"}


# ------------------------------------------------------------------ l'état de la note : des cas
def nouvel_etat() -> Dict:
    """La note neuve de l'écran Streamlit : un cas, la console C3 en méthode Eurocode — nommé court (« S1 » ; l'écran
    Streamlit le nomme « C3 · Console fléchie », que la ligne de l'élément couperait sur une tablette)."""
    e = ET.nouvel_etat()
    e["cas"][0]["nom"] = NOM_DEFAUT
    return e


def normaliser(etat: Dict) -> Dict:
    """L'état reçu (navigateur, fichier .stab, JSON de l'écran Streamlit) relu par le moteur (``etat.depuis_json`` :
    clés manquantes complétées, configuration contrôlée), puis le tableau des cordons libres relu comme l'écran
    Streamlit le relit à chaque affichage. Une note abîmée lève une erreur claire."""
    if not isinstance(etat, dict) or not isinstance(etat.get("cas"), list) or not etat["cas"]:
        raise ValueError("note de soudures illisible : aucun cas")
    if not all(isinstance(c, dict) for c in etat["cas"]):
        raise ValueError("note de soudures abîmée : un cas illisible")
    actif = etat.get("actif", 0)
    if not (isinstance(actif, int) and not isinstance(actif, bool)) and not (isinstance(actif, str) and re.fullmatch(r"-?\d+", actif)):
        raise ValueError(f"note de soudures abîmée : cas ouvert illisible ({actif!r})")
    for c in etat["cas"]:
        for k in ("params", "efforts", "methode"):
            if k in c and not isinstance(c[k], dict):
                raise ValueError(f"note de soudures abîmée : {k} illisible")
    try:
        import json
        out = ET.depuis_json(json.dumps(etat))
        for cas in out["cas"]:
            if not isinstance(cas.get("params"), dict) or not isinstance(cas.get("efforts"), dict) or not isinstance(cas.get("methode"), dict):
                raise ValueError("cas incomplet")
            cas["nom"] = str(cas.get("nom") or "")
            controler_cas(cas)
            _normaliser_c8(cas)
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        raise ValueError(f"note de soudures abîmée : {e}") from None
    return out


def _est_nombre(x) -> bool:
    """Un nombre fini, dans la portée du moteur (``BORNE``)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and abs(x) <= BORNE


def controler_cas(cas: Dict) -> None:
    """Une note venue d'un fichier ne porte que ce que l'écran laisse saisir. L'écran Streamlit, lui, AFFICHE la
    première option d'un menu dont la valeur n'est pas dans la liste (g = 9,8 N/kg se lit « 9,81 ») tout en calculant
    avec la valeur du fichier, ramène en silence un côté ±2 à +1, et s'arrête sur une valeur sous le minimum d'un
    champ : le site refuse la note et dit pourquoi (ValueError)."""
    cfg, p, m = cas["config"], cas["params"], cas["methode"]
    nom = cas.get("nom") or cfg

    def refus(quoi: str):
        raise ValueError(f"{nom} : {quoi}")
    if m.get("mode") not in MODES:
        refus(f"méthode inconnue « {m.get('mode')} »")
    if m.get("ec") not in CRITERES:
        refus(f"critère inconnu « {m.get('ec')} »")
    if not _est_nombre(m.get("g")) or float(m["g"]) not in G_OPTIONS:
        refus(f"g = {m.get('g')} N/kg : l'écran ne propose que 9,81 ou 10")
    for k in ("gamma_M2", "k", "gamma_F"):
        if not _est_nombre(m.get(k)) or m[k] < MINIMUMS[("methode", k)]:
            refus(f"{k} = {m.get(k)} : au moins {C.nb(MINIMUMS[('methode', k)])}")
    for k in NUMERIQUES[cfg]:
        if not _est_nombre(p.get(k)):
            refus(f"{k} = {p.get(k)!r} : un nombre entre −{C.fr(BORNE, 0)} et {C.fr(BORNE, 0)} attendu")
        mini = MINIMUMS.get(("params", k))
        if mini is not None and p[k] < mini:
            refus(f"{k} = {C.nb(p[k])} mm : au moins {C.nb(mini)} mm")

    def troncons_lisibles(tr) -> bool:
        n, lt = tr.get("n", 0), tr.get("l", 60.0)
        return (_est_nombre(n) and 0 <= n <= N_TRONCONS_MAX and n == int(n) and _est_nombre(lt)
                and lt >= MINIMUMS[("troncons", "l")])
    for k in TRONCONS[cfg]:
        tr = p.get(k) or {}
        if not isinstance(tr, dict) or not troncons_lisibles(tr):
            refus(f"tronçons {tr} : n entier de 0 à {N_TRONCONS_MAX} et l ≥ 1 mm")
    for k in ("nuance_A", "nuance_B"):
        if p.get(k, "S235") not in ET.nuances():
            refus(f"nuance inconnue « {p.get(k)} »")
    for k in ("jonction", "jonction_semelles", "jonction_ame"):
        if k in CF.PARAMS_DEFAUT[cfg] and p.get(k) not in CF.JONCTIONS:
            refus(f"jonction inconnue « {p.get(k)} »")
    if cfg == "C1" and p.get("frontaux") not in FRONTAUX:
        refus(f"cordons frontaux : {p.get('frontaux')!r} (0, 1 ou 2)")
    if cfg == "C5" and p.get("semelles") not in SEMELLES:
        refus(f"semelles : « {p.get('semelles')} »")
    if cfg == "C7" and p.get("penetration") not in PENETRATIONS:
        refus(f"pénétration : « {p.get('penetration')} »")
    if cfg in ("C3", "C4") and p.get("poteau", "plat") != "plat" and p.get("poteau") not in _profils("I/H"):
        refus(f"poteau inconnu au catalogue : « {p.get('poteau')} »")
    if cfg in GROUPES_PROFIL and p.get("profil") not in _profils(GROUPES_PROFIL[cfg]):
        refus(f"profilé inconnu au catalogue : « {p.get('profil')} »")
    if cfg == "C4" and any(c not in COTES_C4 for c in p.get("cotes", [])):
        refus(f"côtés : {p.get('cotes')}")
    for k, _, _, _ in CF.EFFORTS[cfg]:
        if not _est_nombre(cas["efforts"].get(k, 0.0)):
            refus(f"effort {k} = {cas['efforts'].get(k)!r} : un nombre entre −{C.fr(BORNE, 0)} et {C.fr(BORNE, 0)} attendu")
    if cfg == "C8":
        els = p.get("elements") if "elements" in p else p.get("cordons")
        if not isinstance(els, list):
            refus("cordons libres illisibles")
        for el in els:
            if not isinstance(el, dict):
                refus("cordon libre illisible")
            nom = str(el.get("nom", ""))
            if el.get("cote", 1) not in (1, -1):
                refus(f"cordon « {nom} » : côté {el.get('cote')!r} (+1 ou −1)")
            for k in COORDS_SEGMENT + COORDS_ARC + ("a",):
                if el.get(k) is not None and not _est_nombre(el[k]):
                    refus(f"cordon « {nom} » : {k} = {el[k]!r} n'est pas un nombre lisible")
            if _est_nombre(el.get("a")) and el["a"] < MINIMUMS[("element", "a")]:
                refus(f"cordon « {nom} » : gorge {C.nb(el['a'])} mm (au moins 1 mm)")
            tr = el.get("troncons") or {}
            if not isinstance(tr, dict) or not troncons_lisibles(tr):
                refus(f"cordon « {nom} » : tronçons {tr} (n entier de 0 à {N_TRONCONS_MAX}, l ≥ 1 mm)")


def _cas(etat: Dict) -> Dict:
    return ET.cas_actif(etat)


def _nom_libre(cas: List[Dict]) -> str:
    """Le nom d'un cas ajouté : « S<n> », le premier que la note ne porte pas."""
    pris = {c.get("nom") for c in cas}
    k = len(cas) + 1
    while f"S{k}" in pris:
        k += 1
    return f"S{k}"


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les cas de la note, comme la barre d'outils de l'écran Streamlit : ouvrir un cas, « ＋ Cas » (une console C3 qui
    garde la méthode du cas ouvert ; le site la nomme court, « S<n> »), copier un cas (``index`` ; sans : le cas ouvert —
    la copie s'insère juste après, l'écran Streamlit la posait en fin de note), en supprimer un (jamais le dernier)."""
    cas = etat["cas"]
    if index is not None and not (isinstance(index, int) and not isinstance(index, bool)) \
            and not (isinstance(index, str) and re.fullmatch(r"-?\d+", index)):
        raise ValueError(f"cas inconnu : {index!r}")
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(cas) - 1)
    elif action == "ajouter":
        c = ET.nouveau_cas("C3", _nom_libre(cas))
        c["methode"] = copy.deepcopy(_cas(etat)["methode"])
        cas.append(c)
        etat["actif"] = len(cas) - 1
    elif action == "dupliquer":
        i = etat["actif"] if index is None else int(index)
        if not 0 <= i < len(cas):
            raise ValueError(f"cas inconnu : {index!r}")
        d = copy.deepcopy(cas[i])
        d["nom"] = f"{d['nom']} (copie)"
        cas.insert(i + 1, d)
        etat["actif"] = i + 1
    elif action == "supprimer" and len(cas) > 1:
        i = int(index or 0)
        if not 0 <= i < len(cas):                       # ramené dans la liste, « 99 » supprimerait le dernier cas
            raise ValueError(f"cas inconnu : {index!r}")
        del cas[i]
        if i == etat["actif"]:
            etat["actif"] = max(0, etat["actif"] - 1)          # comme l'écran Streamlit : le cas d'avant
        elif i < etat["actif"]:
            etat["actif"] -= 1
        etat["actif"] = min(etat["actif"], len(cas) - 1)


# ------------------------------------------------------------------ le tableau des cordons libres (C8)
def lignes_c8(params: Dict) -> List[Dict]:
    """Les lignes du tableau des cordons libres, telles que l'écran Streamlit les pose (``_elements_libres``)."""
    lignes = []
    for el in CF.elements_c8(params):
        tr = el.get("troncons") or {}
        lignes.append({"type": el["type"], "nom": el["nom"], "x1": el.get("x1"), "y1": el.get("y1"), "x2": el.get("x2"), "y2": el.get("y2"),
                       "xc": el.get("xc"), "yc": el.get("yc"), "r": el.get("r"), "ang1": el.get("ang1"), "ang2": el.get("ang2"),
                       "a": el.get("a", 5.0), "cote": int(el.get("cote", 1)), "n": int(tr.get("n", 0) or 0), "l": float(tr.get("l", 60.0) or 60.0)})
    return lignes


def element_depuis_ligne(row: Dict) -> Dict:
    """Une ligne du tableau → un élément, comme ``elements_depuis_lignes`` de l'écran Streamlit : une cellule vide
    (None) prend le défaut du type ; côté ±1, sinon +1 ; retours d'angle « non »."""
    def vide(v):
        return v is None or v != v
    typ = row.get("type") if row.get("type") in ("segment", "arc") else "segment"
    base = dict(CF.ELEMENT_ARC if typ == "arc" else CF.ELEMENT_SEGMENT)
    el = {"type": typ, "nom": (row.get("nom") if isinstance(row.get("nom"), str) else "") or ""}
    for k in COORDS_SEGMENT + COORDS_ARC + ("a",):
        v = row.get(k)
        el[k] = float(v) if not vide(v) else float(base.get(k, 0.0))
    el["cote"] = int(row.get("cote")) if row.get("cote") in (1, -1, 1.0, -1.0) else 1
    el["retours"] = False
    n, lt = row.get("n"), row.get("l")
    el["troncons"] = {"n": int(n) if not vide(n) else 0, "l": float(lt) if not vide(lt) else 60.0}
    return el


def _normaliser_c8(cas: Dict) -> None:
    """Le tableau relu comme l'écran Streamlit le relit à chaque affichage (C8 seulement)."""
    if cas.get("config") != "C8":
        return
    p = cas["params"]
    p["elements"] = [element_depuis_ligne(r) for r in lignes_c8(p)]
    p.pop("cordons", None)


# ------------------------------------------------------------------ la saisie
def _nombre(valeur, mini=None, entier=False, unite="", maxi=None) -> float:
    x = C.lire_nombre(valeur)
    if x is None:
        raise _Refus(f"Nombre illisible : « {valeur} »")
    if abs(x) > BORNE:
        raise _Refus(f"Hors de portée : « {valeur} » (au plus {C.fr(BORNE, 0)} en valeur absolue)")
    if entier and abs(x - round(x)) > 1e-9:
        raise _Refus(f"Nombre entier attendu : « {valeur} »")
    if mini is not None and x < mini - 1e-12:
        raise _Refus(f"Au moins {C.nb(mini)}{(' ' + unite) if unite else ''} : « {valeur} » est refusé (l'écran Streamlit ne l'accepte pas)")
    if maxi is not None and x > maxi + 1e-12:
        raise _Refus(f"Au plus {C.nb(maxi)}{(' ' + unite) if unite else ''} : « {valeur} » est refusé")
    return int(round(x)) if entier else float(x)


class _Refus(Exception):
    """Une saisie refusée : le message part à l'écran, la note reste intacte."""


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans le cas ouvert, avec les effets de l'écran Streamlit. ``chemin`` : les clés de l'état
    (« params », « a »), un indice pour un élément libre ; « + » ajoute un élément libre, « <i>/- » le retire.
    Rend un message si la saisie est refusée (la note reste alors intacte), None sinon."""
    chemin = [int(c) if isinstance(c, str) and re.fullmatch(r"-?\d+", c) else c for c in chemin]
    avant = copy.deepcopy(etat)
    try:
        _ecrire(etat, chemin, valeur)
    except _Refus as e:
        etat.clear()
        etat.update(avant)
        return str(e)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        etat.clear()
        etat.update(avant)
        return f"Saisie refusée : {e}"
    return None


def _ecrire(etat: Dict, chemin: List, valeur) -> None:
    if not chemin:
        raise _Refus("Saisie sans destination")
    tete = chemin[0]
    if tete == "identification":
        if len(chemin) != 2 or chemin[1] not in ("projet", "partie", "indice"):
            raise _Refus("Champ d'identification inconnu")
        etat.setdefault("identification", {"projet": "", "partie": "", "indice": "0"})[chemin[1]] = str(valeur or "")
        return
    if tete == "import":
        _importer(etat, valeur)
        return
    cas = _cas(etat)
    if chemin == ["nom"]:
        cas["nom"] = str(valeur or "")
    elif chemin == ["config"]:
        v = str(valeur)
        if v not in CF.CONFIGS:
            raise _Refus(f"Configuration inconnue : « {v} »")
        if v != cas["config"]:
            ET.changer_config(cas, v)
            _normaliser_c8(cas)
    elif tete == "methode" and len(chemin) == 2:
        _methode(cas, chemin[1], valeur)
    elif tete == "efforts" and len(chemin) == 2:
        cles = [c[0] for c in CF.EFFORTS[cas["config"]]]
        if chemin[1] not in cles:
            raise _Refus(f"Effort inconnu pour {cas['config']} : {chemin[1]}")
        cas["efforts"][chemin[1]] = _nombre(valeur)
    elif tete == "params" and len(chemin) >= 2:
        _params(cas, chemin[1:], valeur)
    else:
        raise _Refus(f"Champ inconnu : {'/'.join(str(c) for c in chemin)}")


def _methode(cas: Dict, k: str, valeur) -> None:
    m = cas["methode"]
    if k == "mode":
        if valeur not in MODES:
            raise _Refus(f"Méthode inconnue : « {valeur} »")
        m["mode"] = valeur
    elif k == "ec":
        if valeur not in CRITERES:
            raise _Refus(f"Critère inconnu : « {valeur} »")
        m["ec"] = valeur
    elif k == "g":
        x = _nombre(valeur)
        if x not in G_OPTIONS:
            raise _Refus("g vaut 9,81 ou 10 N/kg")
        m["g"] = float(x)
    elif k in ("gamma_M2", "k", "gamma_F"):
        m[k] = _nombre(valeur, MINIMUMS[("methode", k)])
    elif k == "k_eurocode":                      # le bouton « Prendre la valeur Eurocode »
        ac = ET.acier_cordon(cas["params"])
        kq = K.k_equivalent_eurocode(ac["fu"], ac["bw"], m["gamma_M2"], m["gamma_F"], m["g"])
        m["k"] = round(kq["k_kg_mm_cm"], 1)
    elif k == "k_bureau":                        # le bouton « Revenir à 100 »
        m["k"] = K.K_BUREAU
    else:
        raise _Refus(f"Champ de méthode inconnu : {k}")


def _profils(groupe: str) -> List[str]:
    """Les profilés d'un groupe du catalogue, dans l'ordre du catalogue (la liste de l'écran Streamlit)."""
    return [n for n, r in CAT.par_nom().items() if FA.FAMILLES.get(r["famille"], {}).get("groupe") == groupe]


def _familles(groupe: str) -> List[str]:
    pres = {r["famille"] for r in CAT.par_nom().values()}
    return [f for f, d in FA.FAMILLES.items() if d.get("groupe") == groupe and f in pres]


def _taille(rec: Dict) -> float:
    return float(rec.get("d") or max(rec.get("h") or 0.0, rec.get("b") or 0.0))


def _voisin(famille: str, ref: Optional[Dict]) -> str:
    """Dans une autre famille, le profilé de taille la plus proche (de masse la plus proche à taille égale)."""
    cands = [r for r in CAT.par_nom().values() if r["famille"] == famille]
    if not cands:
        raise _Refus(f"Famille inconnue : « {famille} »")
    if ref is None:
        ref = CAT.par_nom()[CF.POTEAU_DEFAUT]
    return min(cands, key=lambda r: (abs(_taille(r) - _taille(ref)), abs(r["masse"] - ref["masse"])))["nom"]


def _params(cas: Dict, ch: List, valeur) -> None:
    p = cas["params"]
    cfg = cas["config"]
    k = ch[0]
    if k in TRONCONS[cfg] and len(ch) == 2:
        tr = p.setdefault(k, dict(CF.CONTINU))
        if ch[1] == "on":                        # la case de l'écran Streamlit (_troncons)
            tr["n"] = max(1, int(tr.get("n") or 0)) if bool(valeur) else 0
        elif ch[1] in ("n", "l"):
            tr[ch[1]] = _nombre(valeur, MINIMUMS[("troncons", ch[1])], entier=ch[1] == "n", unite="mm" if ch[1] == "l" else "",
                                maxi=N_TRONCONS_MAX if ch[1] == "n" else None)
        else:
            raise _Refus(f"Champ de tronçons inconnu : {ch[1]}")
        return
    if k == "cotes" and cfg == "C4" and len(ch) == 2:
        if ch[1] not in COTES_C4:
            raise _Refus(f"Côté inconnu : {ch[1]}")
        cur = [c for c in p.get("cotes", []) if c in COTES_C4]
        if bool(valeur) and ch[1] not in cur:
            cur.append(ch[1])
        elif not bool(valeur) and ch[1] in cur:
            cur.remove(ch[1])
        p["cotes"] = cur
        return
    if k == "elements" and cfg == "C8":
        _elements(p, ch[1:], valeur)
        return
    if len(ch) != 1:
        raise _Refus(f"Champ inconnu : {'/'.join(str(c) for c in ch)}")
    if k in NUMERIQUES[cfg]:
        p[k] = _nombre(valeur, MINIMUMS.get(("params", k)), unite="mm")
    elif k in BOOLEENS[cfg]:
        p[k] = bool(valeur)
    elif k in ("nuance_A", "nuance_B"):
        if valeur not in ET.nuances():
            raise _Refus(f"Nuance inconnue : « {valeur} »")
        p[k] = valeur
    elif k in ("jonction", "jonction_semelles", "jonction_ame") and k in CF.PARAMS_DEFAUT[cfg]:
        if valeur not in CF.JONCTIONS:
            raise _Refus(f"Jonction inconnue : « {valeur} »")
        p[k] = valeur
    elif k == "frontaux" and cfg == "C1":
        x = _nombre(valeur, entier=True)
        if x not in FRONTAUX:
            raise _Refus("Cordons frontaux : 0, 1 ou 2")
        p[k] = x
    elif k == "semelles" and cfg == "C5":
        if valeur not in SEMELLES:
            raise _Refus(f"Semelles : « {valeur} » inconnu")
        p[k] = valeur
    elif k == "penetration" and cfg == "C7":
        if valeur not in PENETRATIONS:
            raise _Refus(f"Pénétration : « {valeur} » inconnue")
        p[k] = valeur
    elif k == "poteau" and cfg in ("C3", "C4"):
        if valeur != "plat" and valeur not in _profils("I/H"):
            raise _Refus(f"Poteau inconnu au catalogue : « {valeur} »")
        p[k] = valeur
    elif k == "poteau_famille" and cfg in ("C3", "C4"):    # le premier menu du support : plat, ou une famille I/H
        cur = p.get("poteau") or "plat"
        rec = CAT.par_nom().get(cur)
        if valeur == "plat":
            p["poteau"] = "plat"
        elif valeur not in _familles("I/H"):
            raise _Refus(f"Famille inconnue : « {valeur} »")
        elif rec is None or rec["famille"] != valeur:
            p["poteau"] = _voisin(valeur, rec)
    elif k == "profil" and cfg in GROUPES_PROFIL:
        if valeur not in _profils(GROUPES_PROFIL[cfg]):
            raise _Refus(f"Profilé inconnu au catalogue : « {valeur} »")
        p[k] = valeur
    elif k == "profil_famille" and cfg in GROUPES_PROFIL:
        rec = CAT.par_nom().get(p.get("profil") or "")
        if valeur not in _familles(GROUPES_PROFIL[cfg]):
            raise _Refus(f"Famille inconnue : « {valeur} »")
        if rec is None or rec["famille"] != valeur:
            p["profil"] = _voisin(valeur, rec)
    else:
        raise _Refus(f"Champ inconnu pour {cfg} : {k}")


def _elements(p: Dict, ch: List, valeur) -> None:
    """Les cordons libres : une ligne du tableau de l'écran Streamlit à la fois."""
    lignes = lignes_c8(p)
    if ch == ["+"]:
        lignes.append({"type": valeur if valeur in ("segment", "arc") else "segment"})   # une ligne neuve, cellules vides
    elif len(ch) == 2 and ch[1] == "-":
        if len(lignes) <= 1:
            raise _Refus("Il faut au moins un cordon")
        if not isinstance(ch[0], int) or not 0 <= ch[0] < len(lignes):      # « -1 » retirerait le dernier en silence
            raise _Refus("Cordon inconnu")
        del lignes[ch[0]]
    elif len(ch) >= 2 and isinstance(ch[0], int):
        if not 0 <= ch[0] < len(lignes):
            raise _Refus("Cordon inconnu")
        row = lignes[ch[0]]
        k = ch[1] if len(ch) == 2 else "/".join(ch[1:])
        if k == "type":
            if valeur not in ("segment", "arc"):
                raise _Refus(f"Type inconnu : « {valeur} »")
            if valeur != row["type"]:
                row["type"] = valeur
                # amélioration du site : le nouveau type prend ses valeurs par défaut quand celles de la ligne le rendraient
                # dégénéré (un segment devenu arc a r = 0 dans l'écran Streamlit, qui le refuse ensuite)
                if valeur == "arc" and not (row.get("r") or 0) > 0:
                    row.update({c: CF.ELEMENT_ARC[c] for c in COORDS_ARC})
                if valeur == "segment" and (row.get("x1"), row.get("y1")) == (row.get("x2"), row.get("y2")):
                    row.update({c: CF.ELEMENT_SEGMENT[c] for c in COORDS_SEGMENT})
        elif k == "nom":
            nom = str(valeur or "").strip()
            if nom and any(r["nom"] == nom for i, r in enumerate(lignes) if i != ch[0]):
                raise _Refus(f"Un autre cordon s'appelle déjà « {nom} » : leurs résultats se mêleraient")
            row["nom"] = nom
        elif k == "cote":
            x = _nombre(valeur, entier=True)
            if x not in (1, -1):
                raise _Refus("Côté : +1 ou −1")
            row["cote"] = x
        elif k == "a":
            row["a"] = _nombre(valeur, MINIMUMS[("element", "a")], unite="mm")
        elif k in ("troncons/n", "troncons/l"):
            c = k.split("/")[1]
            row[c] = _nombre(valeur, MINIMUMS[("element", c)], entier=c == "n", unite="mm" if c == "l" else "",
                             maxi=N_TRONCONS_MAX if c == "n" else None)
        elif k in COORDS_SEGMENT + COORDS_ARC:
            row[k] = _nombre(valeur)
        else:
            raise _Refus(f"Champ de cordon inconnu : {k}")
    else:
        raise _Refus("Saisie de cordon illisible")
    p["elements"] = [element_depuis_ligne(r) for r in lignes]
    p.pop("cordons", None)
    p["elements"] = [element_depuis_ligne(r) for r in lignes_c8(p)]          # les noms complétés, comme au rendu suivant


def _importer(etat: Dict, valeur) -> None:
    """Le bouton « Ouvrir » de l'écran Streamlit : un fichier JSON du module (tous ses cas) remplace la note."""
    if not isinstance(valeur, dict) or not isinstance(valeur.get("texte"), str):
        raise _Refus("Fichier illisible")
    try:
        neuf = normaliser(ET.depuis_json(valeur["texte"].lstrip("﻿")))
    except (ValueError, TypeError, KeyError) as e:
        raise _Refus(f"Fichier invalide : {e}") from None
    etat.clear()
    etat.update(neuf)


# ------------------------------------------------------------------ nombres et symboles
def _fr(x: Optional[float], nd: int = 1) -> str:
    """Un résultat à la française (C.fr) avec le signe moins typographique « − », celui des substitutions du moteur
    (« −105,8 » dans la même fenêtre que « (−157,9)² ») ; le trait d'union « - » ne s'y lit pas comme un signe."""
    return C.fr(x, nd).replace("-", "−")


_INDICE_TEXTE = re.compile(r"(?<![A-Za-zÀ-ÿ])([A-Za-zσητγαμβ])_([A-Za-z0-9]+(?:,[A-Za-z0-9]+)*)")


def _texte_indices(texte: str) -> str:
    """Un message du moteur en HTML : échappé, puis ses symboles écrits avec leur indice (« M_x » → M<sub>x</sub>,
    « f_vw,d » → f<sub>vw,d</sub>). Un symbole d'une lettre seulement : un nom saisi (« plat_A ») reste tel quel."""
    return _INDICE_TEXTE.sub(r"\1<sub>\2</sub>", C.esc(texte))


def _fr_moteur(x: float, nd: int) -> str:
    """Un nombre qui côtoie une formule du moteur (tableau et verdict de la fenêtre de calcul) arrondi COMME le moteur
    l'imprime (``acier.formats.F`` : demi vers l'extérieur sur la valeur exacte) : F_adm = 98,1 × 5 = 490,5 N/mm s'écrit
    « 491 » dans la formule ; C.fr (arrondi au pair de Python) l'écrirait « 490 » dans le tableau voisin."""
    s = AF.F(x, nd)
    ent, _, dec = s.partition(",")
    signe = "−" if ent.startswith("-") else ""
    ent = f"{int(ent.lstrip('-')):,}".replace(",", " ")
    return signe + ent + ("," + dec if dec else "")


_TEXTE_SVG = re.compile(r"(<text\b[^>]*>)([^<]*)(</text>)")
_TAILLE_SVG = re.compile(r'font-size="([\d.]+)"')
_INDICE_SVG = re.compile(r"(?<![A-Za-zÀ-ÿ])([A-Za-zσητγαμβ])_([A-Za-z0-9]+(?:,[A-Za-z0-9]+)*)")


def indices_svg(svg: str) -> str:
    """Les symboles des dessins du moteur écrits avec leur indice : « h_p = 200 » → h<tspan>p</tspan> = 200, comme
    les dessins de la poutre (M<sub>inf</sub>) — le moteur, partagé avec l'écran Streamlit et la note, écrit « h_p ».
    Seuls les TEXTES des dessins changent (un symbole d'une lettre suivi de « _ ») : ni un trait, ni une cote, ni un
    nombre ; un nom saisi (« plat_A ») n'est pas touché. Le décalage est en unités du dessin (la taille du texte)."""
    def texte(m):
        corps = m.group(2)
        if "_" not in corps:
            return m.group(0)
        t = _TAILLE_SVG.search(m.group(1))
        fs = float(t.group(1)) if t else 11.0
        dy, petit = round(0.3 * fs, 2), round(0.72 * fs, 2)
        morceaux, pos, decale = [], 0, False
        for mi in _INDICE_SVG.finditer(corps):
            avant = corps[pos:mi.start()] + mi.group(1)
            morceaux.append(f'<tspan dy="{-dy}">{avant}</tspan>' if decale else avant)
            morceaux.append(f'<tspan dy="{dy}" font-size="{petit}">{mi.group(2)}</tspan>')
            decale, pos = True, mi.end()
        reste = corps[pos:]
        if decale and reste:
            morceaux.append(f'<tspan dy="{-dy}">{reste}</tspan>')
        else:
            morceaux.append(reste)
        return m.group(1) + "".join(morceaux) + m.group(3)
    return _TEXTE_SVG.sub(texte, svg)


_SYMBOLES_COLLES = ((re.compile(r"(?<![A-Za-zÀ-ÿ])f([uy])(?![A-Za-zÀ-ÿ])"), r"f_\1"), (re.compile(r"βw"), "β_w"),
                    (re.compile(r"γM([02])"), r"γ_M\1"))


def _encart(svg: str) -> str:
    """L'encart de la gorge (dessin.encart_gorge) : son critère s'écrit « σ_c ≤ fu/(βw·γM2) » ; les symboles collés y
    prennent leur indice comme dans la fenêtre de calcul (f<sub>u</sub>, β<sub>w</sub>, γ<sub>M2</sub>)."""
    def texte(m):
        corps = m.group(2)
        for motif, rempl in _SYMBOLES_COLLES:
            corps = motif.sub(rempl, corps)
        return m.group(1) + corps + m.group(3)
    return indices_svg(_TEXTE_SVG.sub(texte, svg))


def _sym(s: str) -> str:
    """Un symbole du moteur écrit comme dans la note (C.symb_math), β_w et lim_⊥ compris."""
    return C.symb_math(s).replace("β_w", "β<sub>w</sub>").replace("_⊥", "<sub>⊥</sub>")


def _fx(lignes: List[Dict]) -> str:
    """Les lignes « symbole = formule = valeurs = résultat » (C.lignes_fx), les indices grecs du moteur compris."""
    return C.lignes_fx(lignes).replace("β_w", "β<sub>w</sub>").replace("_⊥", "<sub>⊥</sub>")


_PUISSANCE_NEGATIVE = re.compile(r"(?<![\w,)])(−\d[\d  ]*(?:,\d+)?)([²³])")


def _parentheses(texte: str) -> str:
    """« √(0² + −157,9² + 747,9²) » → « √(0² + (−157,9)² + 747,9²) » : le carré porte sur le nombre négatif, comme le
    moteur le calcule (formules.evaluer élève la valeur signée) ; seule l'écriture change."""
    return _PUISSANCE_NEGATIVE.sub(r"(\1)\2", texte)


def _ligne_fx(s) -> Dict:
    parts, r = KF.morceaux(s)
    parts = parts[:1] + [_parentheses(p) for p in parts[1:]]
    if s.nom.startswith("η") and s.nd == -1:
        return {"parts": parts, "res": f"{C.pc(C.pct(s.res, 1))} %", "niveau": C.niveau(s.res)}
    return {"parts": parts, "res": r, "niveau": ""}


def _maj(s: str) -> str:
    return s[:1].upper() + s[1:]


def _pct(t: Optional[float]) -> str:
    return f"{C.pc(C.pct(t))}{C.NBSP}%" if t is not None else "—"


# ------------------------------------------------------------------ les lectures de l'écran Streamlit
def phrase_methode(cas: Dict) -> str:
    """La ligne sous le switch de méthode (``_switch`` de l'écran Streamlit) : ce que la position choisie change."""
    m = cas["methode"]
    ac = ET.acier_cordon(cas["params"])
    if m["mode"] == K.ELS:
        kq = K.k_equivalent_eurocode(ac["fu"], ac["bw"], m["gamma_M2"], m["gamma_F"], m["g"])
        return (f"<b>Efforts de service</b>, sans coefficient : résultante par mm de cordon ≤ k·a avec <b>k = {C.nb(m['k'], 3)} kg/(mm·cm)</b> "
                f"= {C.fr(K.tau_adm_bureau(m['k'], m['g']), 1)} N/mm² sur la gorge — l'Eurocode donnerait f<sub>vw,d</sub>/γ<sub>F</sub> = "
                f"{C.fr(kq['tau_N_mm2'], 1)} N/mm² ({C.fr(kq['k_kg_mm_cm'], 0)} kg/(mm·cm)) pour le {C.esc(ac['nuance_ref'])} (section Méthode).")
    return (f"<b>Efforts de calcul (ELU)</b> : EN 1993-1-8 §4.5.3, méthode {CRITERES[m['ec']].split(' (')[0]}, f<sub>u</sub> et β<sub>w</sub> "
            f"de l'acier de référence, γ<sub>M2</sub> = {C.nb(m['gamma_M2'])}. Le taux selon la méthode ELS du bureau est donné à côté, "
            f"avec les mêmes efforts, à titre indicatif.")


def texte_pleine(cas: Dict, t: Optional[float]) -> str:
    """Le texte de la jonction à pleine pénétration (``_jonction`` de l'écran Streamlit) : la soudure vaut la pièce, et
    la gorge de deux cordons d'angle qui donnerait la même résistance en traction (``criteres.gorge_pleine_resistance``)."""
    if not t:
        return "La soudure vaut la pièce : sa section est vérifiée au métal de base, f<sub>y</sub>/γ<sub>M0</sub> (§4.7.1)."
    ac = ET.acier_cordon(cas["params"])
    gp = K.gorge_pleine_resistance(float(t), ac["fy_min"], ac["fu"], ac["bw"], float(cas["methode"].get("gamma_M2", K.GAMMA_M2)))
    return (f"La soudure vaut la pièce (t = {C.nb(t)} mm) : sa section t × l est vérifiée au métal de base, f<sub>y</sub>/γ<sub>M0</sub> (§4.7.1). "
            f"Pour la même résistance en traction avec deux cordons d'angle, il faudrait a ≥ {C.fr(gp['ratio_dir'], 2)}·t = "
            f"<b>{C.fr(gp['directionnelle'], 1)} mm</b> (directionnelle) ou {C.fr(gp['ratio_simp'], 2)}·t = <b>{C.fr(gp['simplifiee'], 1)} mm</b> "
            f"(simplifiée) de chaque côté.")


def epaisseurs_profil(nom: str) -> Tuple[Optional[float], Optional[float]]:
    """(t_f, t_w) d'un profilé du catalogue — t pour un tube —, comme ``_epaisseurs_profil`` de l'écran Streamlit."""
    try:
        rec = CF._profil(nom)
    except ValueError:
        return None, None
    return float(rec.get("tf", 0) or 0) or None, float(rec.get("tw", rec.get("t", 0)) or 0) or None


def comparaison(res: Dict) -> List[Dict]:
    """Les quatre lectures de la même soudure, mêmes efforts (la carte RÉSULTAT de l'écran Streamlit) : clé, libellé,
    taux, choisie. Vide quand tout le groupe est à pleine pénétration (la pièce gouverne, une seule lecture)."""
    comp = res.get("comparaison") or {}
    if not comp or res.get("pleine"):
        return []
    m = res["methode"]
    kq = res.get("k_equivalent") or {}
    lignes = [("directionnelle", "EC3 directionnelle", m.mode == K.EUROCODE and m.ec == K.DIRECTIONNELLE),
              ("simplifiee", "EC3 simplifiée", m.mode == K.EUROCODE and m.ec == K.SIMPLIFIEE),
              ("bureau", f"ELS bureau k = {m.k:g}", m.mode == K.ELS),
              ("bureau_equiv", f"ELS k Eurocode = {kq.get('k_kg_mm_cm', 0):.0f}", False)]
    return [{"cle": c, "libelle": lib, "taux": comp[c], "choisie": sel} for c, lib, sel in lignes if comp.get(c) is not None]


def nature_chaine(c: Dict) -> str:
    """La colonne « longueur efficace » du tableau des cordons de l'écran Streamlit."""
    return ("pleine pénétration : gorge = t, la pièce" if c.get("pleine") else "pénétration partielle" if c["bout_a_bout"] else
            ("pourtour" if c["ferme"] else ("chaîne (l − 2a)" if c["continu"] else ("retours (l entière)" if c["retours"] else "l − 2a"))))


def message_grave(msg: str) -> bool:
    """Un avertissement (en couleur) plutôt qu'une note, comme l'écran Streamlit les trie."""
    return msg.startswith("Refus") or "DÉPASSÉ" in msg or "ne compte pas" in msg or "ne tiennent pas" in msg


# ------------------------------------------------------------------ les efforts saisis dans le repère du plan de joint
_COMPOSANTES = (("Vx", "V_x"), ("Vy", "V_y"), ("N", "N"), ("Mx", "M_x"), ("My", "M_y"), ("Mz", "M_z"))


def correspondance_axes(cfg: str) -> List[Tuple[str, str, int]]:
    """Chaque effort saisi et la composante du plan de joint que le moteur en fait (``configurations.construire``,
    effort par effort) : (symbole saisi, symbole du plan de joint, signe). Le cadre Efforts nomme les efforts dans le
    repère de la pièce (C5 : V_z le long de l'âme, M_y d'axe fort) ; la vue du plan de joint, les efforts réduits et les
    formules, dans celui du joint (V_y, M_x) — le même effort sous deux noms, que l'écran Streamlit ne relie pas.
    Rien n'est recopié : la correspondance est LUE au moteur."""
    out = []
    cles = [c[0] for c in CF.EFFORTS[cfg]]
    for cle, lib, _, _ in CF.EFFORTS[cfg]:
        try:
            e = CF.construire(cfg, copy.deepcopy(CF.PARAMS_DEFAUT[cfg]), {k: (1.0 if k == cle else 0.0) for k in cles})["efforts"]
        except (ValueError, KeyError, TypeError):      # pragma: no cover — les défauts du moteur se construisent
            return []
        comp = [(nom, getattr(e, attr)) for attr, nom in _COMPOSANTES if abs(getattr(e, attr)) > 1e-12]
        if len(comp) == 1:
            out.append((lib.split(" (")[0], comp[0][0], 1 if comp[0][1] > 0 else -1))
    return out


def phrase_axes(cfg: str) -> str:
    """« Dans le plan de joint : V_z saisi devient V_y, … » — seulement les efforts dont le nom ou le signe change."""
    morceaux = [f"{C.symb_math(s)} devient {'−' if signe < 0 else ''}{C.symb_math(j)}" for s, j, signe in correspondance_axes(cfg)
                if s != j or signe < 0]
    if not morceaux:
        return ""
    return " Dans le repère du joint, celui des dessins et des formules : " + ", ".join(morceaux) + "."


# ------------------------------------------------------------------ les gorges : quel paramètre nourrit quel cordon
def _param_gorge(cas: Dict, chaine: Dict) -> Optional[Dict]:
    """Le paramètre de saisie qui fixe la gorge d'un cordon (le sélecteur de gorge le change) ; None pour une pleine
    pénétration (la gorge est l'épaisseur de la pièce)."""
    if chaine.get("pleine"):
        return None
    cfg, p = cas["config"], cas["params"]
    nom = chaine["nom"]
    if cfg in ("C1", "C4"):
        return {"role": "gorge-a", "chemin": ["params", "a"], "sym": "a", "valeur": p["a"], "renfort": 0.0}
    if cfg in ("C2", "C3", "C6"):
        jon = p.get("jonction", "angle")
        if jon == "partielle":
            return {"role": "gorge-p", "chemin": ["params", "p"], "sym": "p", "valeur": p["p"], "renfort": float(p.get("renfort") or 0)}
        if jon == "angle":
            return {"role": "gorge-a", "chemin": ["params", "a"], "sym": "a", "valeur": p["a"], "renfort": 0.0}
        return None
    if cfg == "C5":
        sem = nom.startswith("semelle")
        jon = p.get("jonction_semelles" if sem else "jonction_ame", "angle")
        suf = "f" if sem else "w"
        if jon == "partielle":
            return {"role": f"gorge-p_{suf}", "chemin": ["params", f"p_{suf}"], "sym": f"p_{suf}", "valeur": p[f"p_{suf}"],
                    "renfort": float(p.get(f"renfort_{suf}") or 0)}
        if jon == "angle":
            return {"role": f"gorge-a_{suf}", "chemin": ["params", f"a_{suf}"], "sym": f"a_{suf}", "valeur": p[f"a_{suf}"], "renfort": 0.0}
        return None
    if cfg == "C7":
        return {"role": "gorge-p", "chemin": ["params", "p"], "sym": "p", "valeur": p["p"], "renfort": 0.0} if p.get("penetration") == "partielle" else None
    if cfg == "C8":
        for i, el in enumerate(CF.elements_c8(p)):
            if nom == el["nom"] or nom.startswith(el["nom"] + " · tronçon "):
                return {"role": f"gorge-el{i}", "chemin": ["params", "elements", i, "a"], "sym": "a", "valeur": el["a"], "renfort": 0.0}
    return None


def _info_role(cas: Dict, res: Dict, role: str) -> Optional[Dict]:
    """Le paramètre d'un sélecteur de gorge et les cordons qu'il nourrit."""
    out = None
    for ch in res.get("chaines", []):
        info = _param_gorge(cas, ch)
        if info and info["role"] == role:
            out = out or dict(info, chaines=[])
            out["chaines"].append(ch["nom"])
    if out is None and role.startswith("gorge-"):           # le paramètre existe même si ses cordons ne comptent plus
        cle = role[len("gorge-"):]
        if cle.startswith("el") and cas["config"] == "C8":
            i = int(cle[2:])
            el = CF.elements_c8(cas["params"])[i]
            return {"role": role, "chemin": ["params", "elements", i, "a"], "sym": "a", "valeur": el["a"], "renfort": 0.0, "chaines": [], "el": el["nom"]}
        if cle in cas["params"]:
            return {"role": role, "chemin": ["params", cle], "sym": cle, "valeur": cas["params"][cle], "renfort": 0.0, "chaines": []}
    return out


def _ecrire_gorge(cas: Dict, chemin: List, v: float) -> Dict:
    c = copy.deepcopy(cas)
    if chemin[1] == "elements":
        lignes = lignes_c8(c["params"])
        lignes[chemin[2]]["a"] = v
        c["params"]["elements"] = [element_depuis_ligne(r) for r in lignes]
    else:
        c["params"][chemin[1]] = v
    return c


def _taux_de(res: Dict, chaines: Sequence[str], el: Optional[str] = None) -> Optional[float]:
    """Le taux des cordons d'un paramètre (le plus chargé), None si le calcul est impossible."""
    if res["statut"] == ET.REFUS:
        return None
    ts = [c["taux"] for c in res["chaines"] if c["nom"] in chaines or (el and (c["nom"] == el or c["nom"].startswith(el + " · tronçon ")))]
    return max(ts) if ts else res["taux_max"]


def candidats_gorge(cas: Dict, info: Dict, jusqu_au_premier: bool = False) -> List[Dict]:
    """Chaque gorge proposée, recalculée par le moteur : taux des cordons qu'elle nourrit, aire des gorges A_w, cordons
    exclus. ``jusqu_au_premier`` : on s'arrête à la plus petite qui tient (l'étoile)."""
    base = PENETRATIONS_P if info["sym"].startswith("p") else GORGES
    vals = sorted(set(base) | {float(info["valeur"])})
    out = []
    for v in vals:
        r = ET.verifier(_ecrire_gorge(cas, info["chemin"], v))
        t = _taux_de(r, info["chaines"], info.get("el"))
        tient = t is not None and t <= 1.0 and not r.get("cordons_exclus")
        out.append({"valeur": v, "taux": t, "A_w": (r.get("proprietes") or {}).get("A_w"), "exclus": bool(r.get("cordons_exclus")), "tient": tient,
                    "gouvernant": r.get("gouvernant")})
        if jusqu_au_premier and tient:
            break
    return out


def gorge_optimale(cas: Dict, info: Dict) -> Optional[Dict]:
    """La plus petite gorge proposée qui tient, tous les cordons comptant (l'étoile ★)."""
    return next((c for c in candidats_gorge(cas, info, True) if c["tient"]), None)


# ------------------------------------------------------------------ le cadre Données
_BALISES = re.compile(r"<[^>]+>")


def _texte(h: str) -> str:
    """Le TEXTE d'un fragment HTML du module (une phrase de l'écran, des indices <sub>) : pour une aide au survol, une
    ligne de l'en-tête d'une vue."""
    return html.unescape(_BALISES.sub("", h))


def _grp(titre: str, aide: str = "") -> str:
    """Un titre de groupe dans une rubrique ; son aide (du TEXTE) au survol de ⓘ, plus en paragraphe."""
    a = f'<span class="aide" title="{C.esc(aide)}">{C.icon("info")}</span>' if aide else ""
    return f'<div class="grp">{titre}{a}</div>'


def _troncons_html(cas: Dict, cle: str, titre: str) -> str:
    tr = cas["params"].get(cle) or dict(CF.CONTINU)
    on = int(tr.get("n") or 0) >= 1
    out = C.case(["params", cle, "on"], on, titre)
    if on:
        out += (f'<div class="sdr-deux">{C.ligne("tronçons", C.num(["params", cle, "n"], tr["n"], "", entier=True, mini=1, maxi=N_TRONCONS_MAX), "n", TRONCONS_AIDE)}'
                f'{C.ligne("longueur", C.num(["params", cle, "l"], tr["l"], "mm", mini=1), "l<sub>t</sub>")}</div>')
    return out


def _jonction_html(cas: Dict, cle: str, cle_a: str, cle_p: str, cle_r: str, t: Optional[float], indice: str = "") -> str:
    p = cas["params"]
    jon = p.get(cle, "angle")
    sub = f"<sub>{indice}</sub>" if indice else ""
    out = C.ligne("Jonction", C.segments(["params", cle], jon, [("angle", "Angle"), ("partielle", "Partielle"), ("pleine", "Pleine")]),
                  full=True, aide=JONCTION_AIDE)
    if jon == "angle":
        out += C.ligne("gorge", C.num(["params", cle_a], p.get(cle_a), "mm", mini=MINIMUMS[("params", cle_a)]), f"a{sub}")
    elif jon == "partielle":
        out += (C.ligne("pénétration", C.num(["params", cle_p], p.get(cle_p), "mm", mini=MINIMUMS[("params", cle_p)]), f"p{sub}")
                + C.ligne("renfort", C.num(["params", cle_r], p.get(cle_r), "mm", mini=0), f"a<sub>r{(',' + indice) if indice else ''}</sub>",
                          "Pénétration par face p ; cordon de renfort a_r, 0 : sans renfort. Cordon d'angle de renfort sur la pénétration : cordon à pénétration profonde de gorge p + a_r (§4.5.2(3))."))
    else:
        out += C.aide(texte_pleine(cas, t))
    return out


def _support_html(cas: Dict) -> str:
    """Le support : un plat, ou l'aile d'un poteau du catalogue (deux menus : la famille, puis la taille)."""
    cur = cas["params"].get("poteau") or "plat"
    rec = CAT.par_nom().get(cur)
    fam = rec["famille"] if rec else "plat"
    out = C.ligne("Support", C.choix(["params", "poteau_famille"], fam, [("plat", "plat (face d'appui plane)")]
                                     + [(f, FA.FAMILLES[f]["libelle"]) for f in _familles("I/H")]), full=True,
                  aide="L'aile d'un poteau du catalogue (dessin, épaisseur t_f pour l'espacement des tronçons) ou un simple plat.")
    if rec:
        out += C.ligne("Poteau", C.choix(["params", "poteau"], cur, [(n, n) for n, r in CAT.par_nom().items() if r["famille"] == fam]), full=True)
    return out


def _profil_html(cas: Dict, titre: str) -> str:
    cfg = cas["config"]
    cur = cas["params"].get("profil")
    rec = CAT.par_nom().get(cur)
    fam = rec["famille"] if rec else None
    out = C.ligne("Famille", C.choix(["params", "profil_famille"], fam, [(f, FA.FAMILLES[f]["libelle"]) for f in _familles(GROUPES_PROFIL[cfg])]),
                  full=True)
    out += C.ligne(titre, C.choix(["params", "profil"], cur, [(n, n) for n, r in CAT.par_nom().items() if r["famille"] == fam] or [(cur, cur)]),
                   full=True)
    if rec:
        dims = (f"d × t = {C.nb(rec['d'])} × {C.nb(rec['t'])} mm" if rec.get("d") else
                f"h × b = {C.nb(rec['h'])} × {C.nb(rec['b'])} mm · " + (f"t = {C.nb(rec['t'])} mm" if rec.get("t") else
                                                                   f"t<sub>w</sub> = {C.nb(rec['tw'])} · t<sub>f</sub> = {C.nb(rec['tf'])} mm"))
        out += C.aide(f"{dims} · {C.fr(rec['masse'], 1)} kg/m · {C.esc(FA.FAMILLES[rec['famille']].get('norme', ''))}")
    return out


def _num(cas: Dict, k: str, unite: str = "mm") -> str:
    return C.num(["params", k], cas["params"].get(k), unite, mini=MINIMUMS.get(("params", k)))


def _geometrie_et_soudure(cas: Dict) -> Tuple[str, str, str, str]:
    """Les sections Géométrie et Soudure de la configuration ouverte, avec leurs résumés."""
    cfg, p = cas["config"], cas["params"]
    L = C.ligne
    retours = C.case(["params", "retours"], bool(p.get("retours")), "Retours d'angle exécutés")
    aide_ret = C.aide("Sans retours d'angle : longueur efficace l − 2a (§4.5.1).")
    if cfg == "C1":
        geo = (L("largeur plat B", _num(cas, "b"), "b") + L("recouvrement", _num(cas, "L_rec"), "L")
               + L("épaisseur plat B", _num(cas, "tB"), "t<sub>B</sub>") + L("épaisseur plat A", _num(cas, "tA"), "t<sub>A</sub>"))
        so = (L("gorge", _num(cas, "a"), "a") + retours + aide_ret
              + C.case(["params", "lateraux"], bool(p.get("lateraux", True)), "Cordons latéraux (le long de N)")
              + L("Frontaux", C.choix(["params", "frontaux"], int(p.get("frontaux", 1)), list(FRONTAUX.items())), full=True,
                  aide="Cordons frontaux (en travers de N)")
              + _troncons_html(cas, "troncons", "Cordons latéraux par tronçons"))
        rg = f"b {C.nb(p['b'])} · L {C.nb(p['L_rec'])} · t {C.nb(p['tB'])} / {C.nb(p['tA'])}"
        rs = f"a {C.nb(p['a'])} · frontaux {int(p.get('frontaux', 1))}"
    elif cfg in ("C2", "C3"):
        if cfg == "C2":
            t = float(p.get("tB", 0) or 0)
            geo = (L("longueur", _num(cas, "l"), "l", "Longueur de la jonction") + L("pièce debout", _num(cas, "tB"), "t<sub>B</sub>", "Épaisseur de la pièce debout B")
                   + L("support", _num(cas, "tA"), "t<sub>A</sub>", "Épaisseur du support A"))
            rg = f"l {C.nb(p['l'])} · t {C.nb(p['tB'])} / {C.nb(p['tA'])}"
        else:
            t = float(p.get("t", 0) or 0)
            geo = (L("hauteur du plat", _num(cas, "h"), "h") + L("épaisseur", _num(cas, "t"), "t")
                   + L("excentricité", _num(cas, "e"), "e", "Excentricité de V : distance de la face du poteau au point d'application de V.")
                   + _support_html(cas))
            rg = f"h {C.nb(p['h'])} × t {C.nb(p['t'])} · e {C.nb(p['e'])} · {C.esc(p.get('poteau') or 'plat')}"
        jon = p.get("jonction", "angle")
        so = (C.case(["params", "deux_cordons"], bool(p.get("deux_cordons", True)), "Soudée des deux faces") + retours
              + _jonction_html(cas, "jonction", "a", "p", "renfort", t))
        if jon != "pleine":
            so += _troncons_html(cas, "troncons", "Soudure discontinue (par tronçons)")
        rs = _resume_jonction(p, "jonction", "a", "p", "renfort") + (" · deux faces" if p.get("deux_cordons", True) else " · une face")
    elif cfg == "C4":
        geo = (L("longueur", _num(cas, "bx"), "b<sub>x</sub>") + L("hauteur", _num(cas, "by"), "b<sub>y</sub>")
               + L("épaisseur", _num(cas, "t"), "t")
               + L("abscisse de V", _num(cas, "xV"), "x<sub>V</sub>", "Point d'application de V, depuis le coin bas-gauche du plat.")
               + L("ordonnée de V", _num(cas, "yV"), "y<sub>V</sub>") + _support_html(cas))
        cotes = [c for c in p.get("cotes", []) if c in COTES_C4]
        so = (L("gorge", _num(cas, "a"), "a") + '<div class="grp">Côtés soudés</div>'
              + '<div class="sdr-cotes">' + "".join(C.case(["params", "cotes", c], c in cotes, _maj(c)) for c in COTES_C4) + "</div>"
              + retours + aide_ret + _troncons_html(cas, "troncons", "Soudure discontinue (par tronçons)"))
        rg = f"{C.nb(p['bx'])} × {C.nb(p['by'])} × {C.nb(p['t'])} · V en ({C.nb(p['xV'])} ; {C.nb(p['yV'])})".replace("-", "−")
        rs = f"a {C.nb(p['a'])} · " + (", ".join(cotes) or "aucun côté")
    elif cfg == "C5":
        tf, tw = epaisseurs_profil(p.get("profil", ""))
        geo = _profil_html(cas, "Profilé") + L("épaisseur platine", _num(cas, "tp"), "t<sub>p</sub>")
        so = (L("Semelles", C.segments(["params", "semelles"], p.get("semelles"), [("deux faces", "Deux faces"), ("extérieur", "Extérieur seul")]),
                full=True, aide="Semelles soudées : deux faces (extérieure et intérieure) ou la face extérieure seule.")
              + C.case(["params", "ame"], bool(p.get("ame", True)), "Âme soudée (les deux faces)") + retours
              + '<div class="grp">Semelles</div>' + _jonction_html(cas, "jonction_semelles", "a_f", "p_f", "renfort_f", tf, "f"))
        if p.get("jonction_semelles", "angle") != "pleine":
            so += _troncons_html(cas, "troncons_semelles", "Semelles par tronçons")
        if p.get("ame", True):
            so += '<div class="grp">Âme</div>' + _jonction_html(cas, "jonction_ame", "a_w", "p_w", "renfort_w", tw, "w")
            if p.get("jonction_ame", "angle") != "pleine":
                so += _troncons_html(cas, "troncons_ame", "Âme par tronçons")
        rg = f"{C.esc(p.get('profil', ''))} · platine {C.nb(p['tp'])} mm"
        rs = ("semelles " + _resume_jonction(p, "jonction_semelles", "a_f", "p_f", "renfort_f")
              + (" · âme " + _resume_jonction(p, "jonction_ame", "a_w", "p_w", "renfort_w") if p.get("ame", True) else " · âme non soudée"))
    elif cfg == "C6":
        _, tt = epaisseurs_profil(p.get("profil", ""))
        geo = _profil_html(cas, "Tube") + L("épaisseur platine", _num(cas, "tp"), "t<sub>p</sub>")
        so = _jonction_html(cas, "jonction", "a", "p", "renfort", tt)
        if p.get("jonction", "angle") != "pleine":
            so += _troncons_html(cas, "troncons", "Pourtour par tronçons (pas de cordon partout)")
        rg = f"{C.esc(p.get('profil', ''))} · platine {C.nb(p['tp'])} mm"
        rs = "pourtour · " + _resume_jonction(p, "jonction", "a", "p", "renfort")
    elif cfg == "C7":
        geo = L("largeur", _num(cas, "b"), "b") + L("épaisseur", _num(cas, "t"), "t")
        pen = p.get("penetration", "pleine")
        so = L("Type", C.segments(["params", "penetration"], pen, [("pleine", "Pleine"), ("partielle", "Partielle")]), full=True,
               aide="Pénétration pleine : la pièce gouverne ; partielle : cordon de gorge égale à la pénétration p.")
        if pen == "partielle":
            so += (L("pénétration", _num(cas, "p"), "p", "Pénétration par face")
                   + C.case(["params", "deux_faces"], bool(p.get("deux_faces", True)), "Soudée des deux faces"))
        else:
            so += C.aide("Pleine pénétration : la soudure vaut le plat, sa section b × t est vérifiée au métal de base, f<sub>y</sub>/γ<sub>M0</sub> (§4.7.1).")
        rg = f"b {C.nb(p['b'])} × t {C.nb(p['t'])}"
        rs = "pleine pénétration" if pen == "pleine" else f"partielle p {C.nb(p['p'])}" + (" · deux faces" if p.get("deux_faces", True) else "")
    else:
        geo, so = _elements_html(cas), ""
        els = CF.elements_c8(p)
        rg = f"{len(els)} cordon(s) · R ({C.nb(p.get('xR'))} ; {C.nb(p.get('yR'))} ; {C.nb(p.get('zR'))})".replace("-", "−")
        rs = ""
    return geo, rg, so, rs


def _resume_jonction(p: Dict, cle: str, cle_a: str, cle_p: str, cle_r: str) -> str:
    jon = p.get(cle, "angle")
    if jon == "angle":
        return f"angle a {C.nb(p.get(cle_a))}"
    if jon == "partielle":
        return f"partielle p {C.nb(p.get(cle_p))}" + (f" + a<sub>r</sub> {C.nb(p.get(cle_r))}" if float(p.get(cle_r) or 0) > 0 else "")
    return "pleine pénétration"


def _elements_html(cas: Dict, res: Optional[Dict] = None) -> str:
    """Les cordons libres (C8) : une carte par ligne du tableau de l'écran Streamlit, puis le point de référence."""
    p = cas["params"]
    els = CF.elements_c8(p)
    out = _grp("Segments et arcs", C8_AIDE.replace("**", ""))
    cartes = ""
    for i, el in enumerate(els):
        tr = el.get("troncons") or {}
        retirer = (f'<button type="button" class="moins" data-chemin="{C._attr_chemin(["params", "elements", i, "-"])}" data-valeur="1" '
                   f'title="Retirer ce cordon">{C.icon("x")}</button>') if len(els) > 1 else ""
        ch = ["params", "elements", i]

        def champ(k, lab, unite="mm", mini=None, entier=False, chemin=None, maxi=None):
            return (f'<label><span>{lab}</span>{C.num(chemin or ch + [k], (tr.get(k.split("/")[-1]) if "/" in k else el.get(k)), unite, mini=mini, entier=entier, maxi=maxi)}</label>')
        if el["type"] == "arc":
            coords = (champ("xc", "x<sub>c</sub>") + champ("yc", "y<sub>c</sub>") + champ("r", "rayon r")
                      + champ("ang1", "angle de départ", "°") + champ("ang2", "angle d'arrivée", "°"))
        else:
            coords = champ("x1", "x<sub>1</sub>") + champ("y1", "y<sub>1</sub>") + champ("x2", "x<sub>2</sub>") + champ("y2", "y<sub>2</sub>")
        cartes += (f'<div class="secard sdr-el"><div class="sh2"><span class="indice">{i + 1}</span>{C.esc(el["nom"])}'
                   f'<span class="muted">· {"arc" if el["type"] == "arc" else "segment"}</span>{retirer}</div>'
                   + C.ligne("Type", C.segments(ch + ["type"], el["type"], [("segment", "Segment"), ("arc", "Arc")]), full=True)
                   + C.ligne("Nom", C.texte(ch + ["nom"], el["nom"], "ex. cordon 1"), full=True)
                   + f'<div class="sdr-grille">{coords}'
                   + champ("a", "gorge a", mini=MINIMUMS[("element", "a")])
                   + champ("troncons/n", "tronçons n", "", mini=0, entier=True, chemin=ch + ["troncons", "n"], maxi=N_TRONCONS_MAX)
                   + champ("troncons/l", "longueur tronçon", "mm", mini=1, chemin=ch + ["troncons", "l"]) + "</div>"
                   + C.ligne("Côté", C.segments(ch + ["cote"], int(el.get("cote", 1)), [(1, "+1 · pièce à gauche"), (-1, "−1 · à droite")]), full=True)
                   + "</div>")
    out += (f'<div class="secards">{cartes}</div><div class="sdr-ajouts">'
            f'<button type="button" class="addrow" data-chemin="params/elements/+" data-valeur="segment">{C.icon("plus")}Ajouter un segment</button>'
            f'<button type="button" class="addrow" data-chemin="params/elements/+" data-valeur="arc">{C.icon("plus")}Ajouter un arc</button></div>'
            '<div class="grp">Point de référence R</div>'
            + C.ligne("x de R", _num(cas, "xR"), "x<sub>R</sub>") + C.ligne("y de R", _num(cas, "yR"), "y<sub>R</sub>")
            + C.ligne("z de R", _num(cas, "zR"), "z<sub>R</sub>",
                      "Hors du plan. Une force dans le plan appliquée à z ≠ 0 crée un moment de flexion (transport en G)."))
    return out


def formulaire(etat: Dict, res: Dict) -> str:
    """Le cadre Données : les rubriques de l'accordéon (la première seule ouverte), chaque champ lié à un chemin de l'état.
    Le nom du cas se modifie sur sa ligne, en tête de la colonne ; ce que fait l'assemblage et ce que change la méthode
    se lisent au survol de ⓘ ; les réglages de la méthode et les fichiers sont des rubriques secondaires."""
    cas = _cas(etat)
    cfg, m, p = cas["config"], cas["methode"], cas["params"]
    L = C.ligne
    # 1 · l'assemblage et la méthode (le switch)
    # (« Type », et non « Assemblage » : avec son icône ⓘ, le libellé long était coupé — mesuré dans le navigateur)
    s1 = (L("Type", C.choix(["config"], cfg, [(c, f"{c} · {d['libelle']}") for c, d in CF.CONFIGS.items()]), full=True,
            aide=CF.CONFIGS[cfg]["description"])
          + L("Méthode", C.segments(["methode", "mode"], m["mode"], [(K.EUROCODE, "Eurocode (ELU)"), (K.ELS, "ELS (bureau)")]), full=True,
              aide=_texte(phrase_methode(cas))))
    geo, rg, so, rs = _geometrie_et_soudure(cas)
    # les efforts
    nature = ("<b>Efforts de service</b> (méthode ELS du bureau : sans coefficient de charge)" if m["mode"] == K.ELS
              else "<b>Efforts de calcul à l'ELU</b> (γ<sub>G</sub>·G + γ<sub>Q</sub>·Q)")
    s_ef = C.aide(f"{nature}, dans le sens du dessin ; une valeur négative retourne la flèche.")
    for cle, lib, unite, aide in CF.EFFORTS[cfg]:
        sym = C.symb_math(lib.split(" (")[0])
        s_ef += L(EFFORTS_COURTS[cfg].get(cle, cle), C.num(["efforts", cle], cas["efforts"].get(cle), unite), sym, aide)
    e = res.get("efforts")
    if e is not None:
        s_ef += C.aide(f"Au point de référence ({_fr(e.reference[0], 0)} ; {_fr(e.reference[1], 0)} ; {_fr(e.reference[2], 0)}) mm, dans le plan "
                       f"de joint : V<sub>x</sub> = {_fr(e.Vx / 1e3)} kN, V<sub>y</sub> = {_fr(e.Vy / 1e3)} kN, N = {_fr(e.N / 1e3)} kN, "
                       f"M<sub>x</sub> = {_fr(e.Mx / 1e6, 2)} kN·m, M<sub>y</sub> = {_fr(e.My / 1e6, 2)} kN·m, M<sub>z</sub> = {_fr(e.Mz / 1e6, 2)} kN·m."
                       + phrase_axes(cfg))
    r_ef = " · ".join(f"{C.symb_math(lib.split(' (')[0])} {C.nb(cas['efforts'].get(cle)).replace('-', '−')}" for cle, lib, _, _ in CF.EFFORTS[cfg]
                      if cas["efforts"].get(cle)) or "aucun effort"
    # les aciers
    ac = ET.acier_cordon(p)
    nz = list(ET.nuances())
    s_ac = (C.ligne2(("Pièce A", C.choix(["params", "nuance_A"], p.get("nuance_A", "S235"), [(n, n) for n in nz]),
                      "Nuance de la pièce A (support). f_u et β_w de référence : la pièce la plus faible (Tab. 4.1)"),
                     ("Pièce B", C.choix(["params", "nuance_B"], p.get("nuance_B", "S235"), [(n, n) for n in nz]), "Nuance de la pièce B (rapportée)"))
            + C.aide(f"Acier de référence <b>{C.esc(ac['nuance_ref'])}</b> (la pièce la plus faible) : f<sub>u</sub> = {C.nb(ac['fu'])} N/mm², "
                     f"β<sub>w</sub> = {C.nb(ac['bw'])} ; f<sub>y</sub> = {C.nb(ac['fy_min'])} N/mm² (pleine pénétration)."))
    # les hypothèses : les réglages des deux méthodes
    d = K.directionnelle(0, 0, 0, ac["fu"], ac["bw"], m["gamma_M2"])
    s = K.simplifiee(0, 1.0, ac["fu"], ac["bw"], m["gamma_M2"])
    kq = K.k_equivalent_eurocode(ac["fu"], ac["bw"], m["gamma_M2"], m["gamma_F"], m["g"])
    s_me = ('<div class="grp">Eurocode · EN 1993-1-8 §4.5.3</div>'
            + L("Critère", C.segments(["methode", "ec"], m["ec"], [(K.DIRECTIONNELLE, "Directionnelle"), (K.SIMPLIFIEE, "Simplifiée")]), full=True,
                aide="Directionnelle : √(σ⊥² + 3(τ⊥² + τ∥²)) ≤ fu/(βw·γM2) et σ⊥ ≤ 0,9·fu/γM2 ; simplifiée : |f| ≤ f_vw,d·a, toujours du côté sûr")
            + L("cordons", C.num(["methode", "gamma_M2"], m["gamma_M2"], "", mini=1), "γ<sub>M2</sub>", "Coefficient partiel des cordons ; 1,25 : NBN EN 1993-1-8 ANB")
            + C.aide(f"Acier de référence {C.esc(ac['nuance_ref'])} : f<sub>u</sub>/(β<sub>w</sub>·γ<sub>M2</sub>) = {C.fr(d['lim_c'], 1)} N/mm², "
                     f"0,9·f<sub>u</sub>/γ<sub>M2</sub> = {C.fr(d['lim_perp'], 1)} N/mm², f<sub>vw,d</sub> = {C.fr(s['f_vw_d'], 1)} N/mm².")
            + '<div class="grp">Méthode ELS du bureau</div>'
            + L("kg/(mm·cm)", C.num(["methode", "k"], m["k"], "", mini=1), "k", "Règle du bureau : k kg par mm de gorge et par cm de cordon")
            + L("Gravité", C.choix(["methode", "g"], float(m["g"]), list(G_OPTIONS.items())), full=True, aide="N par kg")
            + L("ELU / ELS", C.num(["methode", "gamma_F"], m["gamma_F"], "", mini=1), "γ<sub>F</sub>", "Rapport ELU / ELS, pour l'équivalent Eurocode de la règle du bureau.")
            + C.aide(f"k = {C.nb(m['k'], 3)} kg/(mm·cm) = <b>{C.fr(K.tau_adm_bureau(m['k'], m['g']), 1)} N/mm²</b> sur la gorge · l'Eurocode au même "
                     f"niveau de service donne f<sub>vw,d</sub>/γ<sub>F</sub> = <b>{C.fr(kq['tau_N_mm2'], 1)} N/mm² = {C.fr(kq['k_kg_mm_cm'], 0)} kg/(mm·cm)</b> "
                     f"pour le {C.esc(ac['nuance_ref'])}.")
            + '<div class="sdr-ajouts">'
            f'<button type="button" class="addrow" data-chemin="methode/k_eurocode" data-valeur="1">{C.icon("check")}Prendre la valeur Eurocode ({C.fr(kq["k_kg_mm_cm"], 0)})</button>'
            f'<button type="button" class="addrow" data-chemin="methode/k_bureau" data-valeur="1">{C.icon("history")}Revenir à {C.nb(K.K_BUREAU)}</button></div>')
    # la note PDF et le fichier de l'écran Streamlit
    s_fi = (_grp("Identification de la note PDF", NC.AIDE_IDENTIFICATION)
            + C.fichier(["import"], "Ouvrir un fichier JSON de Streamlit", ".json")
            + C.aide("Le fichier « soudures_….json » de l'écran Streamlit (tous ses cas) remplace cette note."))
    r_me = (f"{'ELS · k ' + C.nb(m['k'], 3) if m['mode'] == K.ELS else CRITERES[m['ec']].split(' (')[0]} · γ<sub>M2</sub> {C.nb(m['gamma_M2'])}")
    secs = [C.section(1, "Assemblage", f"{cfg} · {C.esc(CF.CONFIGS[cfg]['libelle'])} · {'ELS' if m['mode'] == K.ELS else 'Eurocode'}", s1, True, "as")]
    if cfg == "C8":
        secs.append(C.section(2, "Cordons libres", rg, geo, False, "c8"))
    else:
        secs.append(C.section(2, "Géométrie", rg, geo, False, "geo"))
        secs.append(C.section(3, "Soudure", rs, so, False, "so"))
    n = len(secs) + 1
    secs.append(C.section(n, "Efforts", r_ef, s_ef, False, "ef"))
    secs.append(C.section(n + 1, "Aciers", f"{C.esc(p.get('nuance_A', 'S235'))} / {C.esc(p.get('nuance_B', 'S235'))}", s_ac, False, "ac"))
    secs.append(C.section(n + 2, "Hypothèses", r_me, s_me, False, "me", secondaire=True))
    secs.append(C.section(n + 3, "Note PDF et reprise", "en-tête de la note · fichier Streamlit", s_fi, False, "fi", secondaire=True))
    return "".join(secs)


def _dims(c: Dict) -> str:
    """« · C3 · Console fléchie » : ce qu'on lit d'un cas dans la liste de la note (HTML)."""
    return f"· {C.esc(c['config'])} · {C.esc(CF.CONFIGS[c['config']]['libelle'])}"


def arbre(etat: Dict, res: Dict) -> str:
    """La colonne de gauche COMPACTE (api_web/etabli_compact.arbre_compact, la même que la poutre) : la ligne du cas
    ouvert (son nom, qu'un clic modifie ; la liste des cas de la note ; copier, supprimer), que l'écran pose dans
    l'en-tête du panneau, puis ses rubriques."""
    noms = [(c.get("nom") or f"S{i + 1}", _dims(c), False) for i, c in enumerate(etat["cas"])]
    return E.arbre(GENRE, noms, etat["actif"], formulaire(etat, res), "cassoudure", compact=True)


# ------------------------------------------------------------------ le résultat
def _chaine_de(res: Dict) -> Dict[str, str]:
    return {c["nom"]: c["chaine"] for c in res.get("cordons", [])}


def _pire_point(res: Dict, nom_chaine: str) -> Optional[Dict]:
    ch = _chaine_de(res)
    pts = [p for p in res.get("points", []) if ch.get(p["cordon"], p["cordon"]) == nom_chaine]
    return max(pts, key=lambda p: p["taux"]) if pts else None


def _piece(ch: Dict, info: Optional[Dict]) -> str:
    """Ce que la ligne d'un cordon dit de lui : sa gorge (ou sa pénétration) — ce que son sélecteur change. La longueur
    efficace est dans la fenêtre de calcul et le tableau des cordons (sur la ligne, elle coupait les noms longs)."""
    if ch.get("pleine"):
        return f"pleine · t = {C.nb(ch['a'])} mm"
    if info and info.get("renfort"):
        return f"p + a<sub>r</sub> = {C.nb(ch['a'])} mm"
    if ch.get("bout_a_bout"):
        return f"p = {C.nb(ch['a'])} mm"
    return f"a = {C.nb(ch['a'])} mm"


def _longueur(ch: Dict) -> str:
    """La longueur efficace d'un cordon, écrite dans le sous-titre de sa fenêtre de calcul."""
    return f"l<sub>eff</sub> = {C.fr(ch['l_eff'], 0)} mm"


_DECIMALE = re.compile(r"(?<![§\d.,])(?<!Tab\. )(?<!fig\. )(\d+)\.(\d+)(?![\d.])")


def texte_fr(msg: str) -> str:
    """Un message du moteur écrit à la française : « 98.1 N/mm² » → « 98,1 N/mm² » (le moteur formate ses nombres
    à l'anglaise, ``:g`` et ``:.1f``) ; les renvois aux articles (§4.5.3.2) ne bougent pas."""
    return _DECIMALE.sub(r"\1,\2", msg)


def _lignes_critere(res: Dict, pt: Dict) -> List[Dict]:
    """Les lignes du tableau d'une fenêtre de calcul : la vérification de la méthode choisie au point, ses deux termes."""
    subs = {s.nom: s for s in FO.critere(res, pt)}
    cle = FO.methode_du_point(res, pt)
    if cle == "directionnelle":
        lig = [{"nom": "Contrainte de comparaison", "formule": "σ<sub>c</sub> = √(σ<sub>⊥</sub>² + 3(τ<sub>⊥</sub>² + τ<sub>∥</sub>²)) ≤ f<sub>u</sub>/(β<sub>w</sub>·γ<sub>M2</sub>)",
                "valeur": f"{_fr_moteur(subs['σ_c'].res, 1)} N/mm²", "limite": f"{_fr_moteur(subs['lim_c'].res, 1)} N/mm²", "t": subs["η_c"].res},
               {"nom": "Contrainte normale", "formule": "|σ<sub>⊥</sub>| ≤ 0,9·f<sub>u</sub>/γ<sub>M2</sub>",
                "valeur": f"{_fr_moteur(abs(pt['sigma_perp']), 1)} N/mm²", "limite": f"{_fr_moteur(subs['lim_⊥'].res, 1)} N/mm²", "t": subs["η⊥"].res}]
    elif cle == "simplifiee":
        lig = [{"nom": "Résistance du cordon", "formule": "|f| ≤ F<sub>w,Rd</sub> = f<sub>vw,d</sub>·a",
                "valeur": f"{_fr_moteur(pt['f'], 0)} N/mm", "limite": f"{_fr_moteur(subs['F_w,Rd'].res, 0)} N/mm", "t": subs["η"].res}]
    elif cle == "els":
        lig = [{"nom": "Résultante de service", "formule": "|f| ≤ F<sub>adm</sub> = τ<sub>adm</sub>·a",
                "valeur": f"{_fr_moteur(pt['f'], 0)} N/mm", "limite": f"{_fr_moteur(subs['F_adm'].res, 0)} N/mm", "t": subs["η"].res}]
    else:
        g = "γ<sub>M0</sub>" if cle == "pleine" else "γ<sub>F</sub>"
        lig = [{"nom": "Métal de base", "formule": f"√(σ² + 3τ²) ≤ f<sub>y</sub>/{g}",
                "valeur": f"{_fr_moteur(subs['σ_c'].res, 1)} N/mm²", "limite": f"{_fr_moteur(subs['lim'].res, 1)} N/mm²", "t": subs["η"].res}]
    tmax = max(li["t"] for li in lig)
    for li in lig:
        li.update(taux=C.pct(li["t"], 1), niveau=C.niveau(li["t"]), gov=li["t"] == tmax and len(lig) > 1)
    return lig


def _verdict_comparaison(res: Dict, g: Dict) -> str:
    """« valeur ≤ limite » de la vérification d'un point (le gouvernant dans le verdict ; le plus chargé d'un cordon sur
    sa ligne), nommée par le moteur (formules.critere) — court : la formule de la limite est dans la fenêtre de calcul."""
    subs = {s.nom: s for s in FO.critere(res, g)}
    cle = FO.methode_du_point(res, g)
    op = "≤" if g["taux"] <= 1 else "&gt;"
    if cle == "directionnelle":
        if subs["η_c"].res >= subs["η⊥"].res:
            return f"σ<sub>c</sub> = {_fr_moteur(subs['σ_c'].res, 1)} {op} {_fr_moteur(subs['lim_c'].res, 1)} N/mm²"
        return f"|σ<sub>⊥</sub>| = {_fr_moteur(abs(g['sigma_perp']), 1)} {op} {_fr_moteur(subs['lim_⊥'].res, 1)} N/mm²"
    if cle == "simplifiee":
        return f"|f| = {_fr_moteur(g['f'], 0)} {op} F<sub>w,Rd</sub> = {_fr_moteur(subs['F_w,Rd'].res, 0)} N/mm"
    if cle == "els":
        return f"|f| = {_fr_moteur(g['f'], 0)} {op} F<sub>adm</sub> = {_fr_moteur(subs['F_adm'].res, 0)} N/mm"
    return f"σ<sub>c</sub> = {_fr_moteur(subs['σ_c'].res, 1)} {op} {_fr_moteur(subs['lim'].res, 1)} N/mm²"


def _hypotheses(res: Dict) -> List[str]:
    m = res["methode"]
    ac = res["acier"]
    out = [f"Acier de référence <b>{C.esc(ac['nuance_ref'])}</b> · f<sub>u</sub> = {C.nb(ac['fu'])} N/mm² · β<sub>w</sub> = {C.nb(ac['bw'])} · "
           f"f<sub>y</sub> = {C.nb(ac['fy_min'])} N/mm²"]
    if m.mode == K.ELS:
        out.append(f"Méthode <b>ELS du bureau</b> · k = {C.nb(m.k, 3)} kg/(mm·cm) · g = {C.nb(m.g)} N/kg · γ<sub>F</sub> = {C.nb(m.gamma_F)}")
        out.append("Efforts <b>de service</b>")
    else:
        out.append(f"Méthode <b>Eurocode {CRITERES[m.ec].split(' (')[0]}</b> · γ<sub>M2</sub> = {C.nb(m.gamma_M2)} · γ<sub>M0</sub> = {C.nb(K.GAMMA_M0)}")
        out.append("Efforts <b>de calcul (ELU)</b>")
    return out


def _efforts_fx(res: Dict) -> str:
    e = res["efforts"]
    lignes = [{"parts": ["R"], "res": f"({_fr(e.reference[0], 0)} ; {_fr(e.reference[1], 0)} ; {_fr(e.reference[2], 0)}) mm", "niveau": ""}]
    for sym, v, u, nd in (("V_x", e.Vx / 1e3, "kN", 2), ("V_y", e.Vy / 1e3, "kN", 2), ("N", e.N / 1e3, "kN", 2),
                          ("M_x", e.Mx / 1e6, "kN·m", 2), ("M_y", e.My / 1e6, "kN·m", 2), ("M_z", e.Mz / 1e6, "kN·m", 2)):
        lignes.append({"parts": [sym], "res": f"{_fr(v, nd)} {u}", "niveau": ""})
    return _fx(lignes)


def _groupe_fx(res: Dict) -> str:
    pr = res.get("proprietes") or {}
    lignes = [_ligne_fx(s) for s in FO.groupe(res)]
    if pr:
        lignes.insert(1, {"parts": ["G"], "res": f"({_fr(pr['xG'], 1)} ; {_fr(pr['yG'], 1)}) mm", "niveau": ""})
        lignes.append({"parts": ["I_xy"], "res": f"{_fr(pr['Ixy'] / 1e4, 1)} cm⁴", "niveau": ""})
        if res.get("beta_Lw", 1.0) < 1.0:
            lignes.append({"parts": ["β_Lw"], "res": _fr(res["beta_Lw"], 3), "niveau": ""})
    return _fx(lignes)


def _autres_lectures(res: Dict, pt: Dict) -> str:
    m = res["methode"]
    kq = res.get("k_equivalent") or {}
    if pt.get("pleine"):
        return ""
    vals = [("EC3 directionnelle", pt["taux_directionnelle"]), ("EC3 simplifiée", pt["taux_simplifiee"]),
            (f"ELS bureau k = {C.nb(m.k, 3)}", pt["taux_bureau"]), (f"ELS k Eurocode = {C.fr(kq.get('k_kg_mm_cm', 0), 0)}", pt["taux_bureau_equiv"])]
    return ('<div class="ck-autres">Au même point, avec les mêmes efforts : '
            + " · ".join(f"{t} <b>{_pct(v)}</b>" for t, v in vals) + "</div>")


def _fenetre_chaine(i: int, ch: Dict, res: Dict, info: Optional[Dict], hyps: List[str]) -> str:
    pt = _pire_point(res, ch["nom"])
    if pt is None:
        return ""
    cle = FO.methode_du_point(res, pt)
    detail = (f'<div class="ck-g"><div class="ck-t">Efforts au point de référence</div>{_efforts_fx(res)}</div>'
              f'<div class="ck-g"><div class="ck-t">Groupe de cordons</div>{_groupe_fx(res)}</div>'
              f'<div class="ck-g"><div class="ck-t">Au point le plus chargé · {C.esc(pt["position"])}</div>{_fx([_ligne_fx(s) for s in FO.point(res, pt)])}</div>'
              f'<div class="ck-g"><div class="ck-t">{C.esc(FO.TITRES[cle])}</div>{_fx([_ligne_fx(s) for s in FO.critere(res, pt)])}</div>')
    lignes = _lignes_critere(res, pt)
    for li in lignes:
        li["sous"] = C.esc(f"{ch['nom']} · {pt['position']}")
    sous = f"{_piece(ch, info)} · {_longueur(ch)} · {C.esc(nature_chaine(ch))} · les formules de la méthode choisie, au point le plus chargé"
    return C.fenetre_calcul(f"c{i}", C.esc(_maj(ch["nom"])), sous, hyps, lignes, detail, _autres_lectures(res, pt))


def _res_methode(res: Dict, cle: str) -> Dict:
    """Le même résultat lu avec une autre méthode (pour écrire ses formules : formules.critere)."""
    m = res["methode"]
    kq = res.get("k_equivalent") or {}
    if cle == "directionnelle":
        mm = K.Methode(K.EUROCODE, K.DIRECTIONNELLE, m.gamma_M2, m.k, m.g, m.gamma_F)
    elif cle == "simplifiee":
        mm = K.Methode(K.EUROCODE, K.SIMPLIFIEE, m.gamma_M2, m.k, m.g, m.gamma_F)
    elif cle == "bureau":
        mm = K.Methode(K.ELS, m.ec, m.gamma_M2, m.k, m.g, m.gamma_F)
    else:
        mm = K.Methode(K.ELS, m.ec, m.gamma_M2, kq.get("k_kg_mm_cm", m.k), m.g, m.gamma_F)
    return dict(res, methode=mm)


def _fenetre_comparaison(res: Dict, hyps: List[str]) -> str:
    lignes, detail = [], ""
    ch = _chaine_de(res)
    for c in comparaison(res):
        pt = max(res["points"], key=lambda p: p[f"taux_{c['cle']}"])
        rm = _res_methode(res, c["cle"])
        lc = _lignes_critere(rm, pt)
        li = max(lc, key=lambda x: x["t"])
        nom = C.esc(ch.get(pt["cordon"], pt["cordon"]))
        lignes.append(dict(li, nom=C.esc(c["libelle"]) + (" ◀" if c["choisie"] else ""), sous=f"{nom} · {C.esc(pt['position'])}", gov=c["choisie"]))
        detail += (f'<div class="ck-g"><div class="ck-t">{C.esc(c["libelle"])} · {nom}, {C.esc(pt["position"])}</div>'
                   f'{_fx([_ligne_fx(s) for s in FO.critere(rm, pt)])}</div>')
    pied = ('<div class="ck-autres">Les quatre lectures appliquent chaque méthode aux MÊMES efforts saisis : la méthode choisie (◀) décide '
            "du statut ; les autres sont indicatives (l'Eurocode attend des efforts de calcul, le bureau des efforts de service).</div>")
    return C.fenetre_calcul("m", "Les quatre lectures de la soudure", "mêmes efforts · chaque méthode à son point le plus chargé", hyps, lignes, detail, pied)


def _tableau_cordons(res: Dict) -> str:
    pr = res.get("proprietes") or {}
    props = ""
    if pr:
        props = ('<div class="sdr-props">'
                 f"<span>A<sub>w</sub> = {_fr(pr.get('A_w', 0), 0)} mm²</span><span>G = ({_fr(pr.get('xG', 0), 1)} ; {_fr(pr.get('yG', 0), 1)}) mm</span>"
                 f"<span>I<sub>x</sub> = {_fr(pr.get('Ix', 0) / 1e4, 1)} cm⁴</span><span>I<sub>y</sub> = {_fr(pr.get('Iy', 0) / 1e4, 1)} cm⁴</span>"
                 f"<span>I<sub>xy</sub> = {_fr(pr.get('Ixy', 0) / 1e4, 1)} cm⁴</span><span>J = {_fr(pr.get('J', 0) / 1e4, 1)} cm⁴</span>"
                 f"<span>{pr.get('n_cordons', 0)} segment(s) de calcul</span><span>β<sub>Lw</sub> = {_fr(res.get('beta_Lw', 1.0), 3)}</span></div>")
    g = res.get("gouvernant") or {}
    ch_g = _chaine_de(res).get(g.get("cordon"), g.get("cordon"))
    rows = "".join(
        f'<tr class="{"gov" if c["nom"] == ch_g else ""}"><td class="nm">{C.esc(c["nom"])}<span>{C.esc(nature_chaine(c))} · le plus chargé : {C.esc(c["position"])}</span></td>'
        f'<td class="n">{_fr(c["a"], 1)}</td><td class="n">{_fr(c["l"], 0)}</td><td class="n">{_fr(c["l_eff"], 0)}</td>'
        f'<td class="n">{_fr(c["f_max"], 0)}</td><td class="n">{C.chip(C.pct(c["taux"]), C.niveau(c["taux"]))}</td></tr>' for c in res.get("chaines", []))
    exclus = ""
    if res.get("cordons_exclus"):
        noms = sorted({c["chaine"] for c in res["cordons_exclus"]})
        exclus = C.aide("Ne comptent pas (gorge &lt; 3 mm ou l − 2a &lt; max(30 mm ; 6a)) : " + C.esc(", ".join(noms)))
    return (f'<div class="sdr-vue">{props}<table class="vtab sdr-tab"><thead><tr><th>Cordon · longueur efficace</th><th class="n">a mm</th>'
            f'<th class="n">l mm</th><th class="n">l<sub>eff</sub> mm</th><th class="n">|f|<sub>max</sub> N/mm</th><th class="n">η</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>{exclus}</div>')


def _tableau_points(res: Dict) -> str:
    r = res.get("resistances") or {}
    lim = (f"Par cordon et par point (début, milieu, fin de la longueur efficace) : force par unité de longueur |f| (N/mm), composantes sur la gorge (N/mm²), "
           f"σ<sub>c</sub> = √(σ<sub>⊥</sub>² + 3(τ<sub>⊥</sub>² + τ<sub>∥</sub>²)), et le taux selon chaque méthode. Limites : "
           f"f<sub>u</sub>/(β<sub>w</sub>·γ<sub>M2</sub>) = {_fr(r.get('lim_c', 0), 1)}, 0,9·f<sub>u</sub>/γ<sub>M2</sub> = {_fr(r.get('lim_perp', 0), 1)}, "
           f"f<sub>vw,d</sub> = {_fr(r.get('f_vw_d', 0), 1)} N/mm² · bureau {_fr(r.get('tau_adm_bureau', 0), 1)} N/mm²"
           + (f" · pleine pénétration (la pièce) : f<sub>y</sub>/γ<sub>M0</sub> = {_fr(r.get('lim_piece', 0), 0)} N/mm², en service "
              f"f<sub>y</sub>/γ<sub>F</sub> = {_fr(r.get('lim_piece_els', 0), 1)}." if res.get("n_pleine") else "."))
    ch = _chaine_de(res)
    g = res.get("gouvernant")

    def chip(t):
        return C.chip(C.pct(t), C.niveau(t))
    rows = "".join(
        f'<tr class="{"gov" if p is g else ""}"><td class="nm">{C.esc(ch.get(p["cordon"], p["cordon"]))}<span>{C.esc(p["position"])}</span></td>'
        f'<td class="n">{_fr(p["f"], 0)}</td><td class="n">{_fr(p["sigma_perp"], 1)}</td><td class="n">{_fr(p["tau_perp"], 1)}</td>'
        f'<td class="n">{_fr(p["tau_par"], 1)}</td><td class="n">{_fr(p["sigma_c"], 1)}</td><td class="n">{chip(p["taux_directionnelle"])}</td>'
        f'<td class="n">{chip(p["taux_simplifiee"])}</td><td class="n">{chip(p["taux_bureau"])}</td><td class="n">{chip(p["taux"])}</td></tr>'
        for p in res.get("points", []))
    return (f'<div class="sdr-vue"><p class="sdr-note">{lim}</p><table class="vtab sdr-tab sdr-pts"><thead><tr><th>Cordon · point</th>'
            f'<th class="n">|f| N/mm</th><th class="n">σ<sub>⊥</sub></th><th class="n">τ<sub>⊥</sub></th><th class="n">τ<sub>∥</sub></th>'
            f'<th class="n">σ<sub>c</sub></th><th class="n">η dir.</th><th class="n">η simpl.</th><th class="n">η ELS</th><th class="n">η choisie</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


def _vue_methode() -> str:
    out = ""
    for sec in MT.SECTIONS:
        out += f"<h4>{C.esc(sec['titre'])}</h4>" + "".join(f"<p>{C.esc(p)}</p>" for p in sec["paragraphes"])
    out += "<h4>Références</h4><ul>" + "".join(f"<li>{C.esc(r)}</li>" for r in MT.REFERENCES) + "</ul>"
    out += f"<h4>Lire les dessins</h4><p>{C.esc(DS.legende())}</p>"
    return f'<div class="sdr-vue sdr-texte">{out}</div>'


def _res_echappe(res: Dict) -> Dict:
    """Une copie du résultat dont les noms de cordon sont échappés : l'encart de la gorge (dessin.encart_gorge) les
    écrit tels quels dans son SVG (l'écran Streamlit aussi : défaut écrit dans la doc)."""
    r = dict(res)
    r["cordons"] = [dict(c, chaine=C.esc(c["chaine"]), nom=C.esc(c["nom"])) for c in res.get("cordons", [])]
    if res.get("gouvernant"):
        r["gouvernant"] = dict(res["gouvernant"], cordon=C.esc(res["gouvernant"]["cordon"]))
    return r


def vues(res: Dict) -> List[Dict]:
    """La toile : les vues du moteur (schemas.vues : deux ou trois par configuration), l'encart de la gorge, les
    tableaux des cordons et des points, la méthode."""
    cfg = res["config"]
    lib = f"{cfg} · {CF.CONFIGS[cfg]['libelle']}"
    out = []
    try:
        dessins = SC.vues(res)
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError):     # pragma: no cover — un dessin impossible n'arrête pas le calcul
        dessins = []
    for k, titre, d in dessins:
        out.append({"cle": k, "titre": titre, "court": VUES_COURTES.get(titre, titre), "svg": indices_svg(SC.svg(d, f"sdr_{k}")), "legende": lib})
    if res.get("gouvernant"):
        # l'encart se borne à 320 px pour la colonne de l'écran Streamlit ; sur la feuille, il prend la place du dessin
        encart = _encart(DS.encart_gorge(_res_echappe(res)).replace('style="max-width:320px;display:block"', 'style="display:block"', 1))
        out.append({"cle": "gorge", "titre": "La gorge au point gouvernant", "court": "Gorge", "svg": encart,
                    "legende": "EN 1993-1-8 fig. 4.5 · σ⊥, τ⊥, τ∥"})
    if res.get("chaines"):
        out.append({"cle": "cordons", "titre": "Les cordons", "court": "Cordons", "svg": _tableau_cordons(res), "legende": "propriétés du groupe, longueurs efficaces, taux"})
    if res.get("points"):
        out.append({"cle": "points", "titre": "Détail par point", "court": "Points", "svg": _tableau_points(res),
                    "legende": "début, milieu et fin de chaque cordon · les quatre lectures"})
    out.append({"cle": "methode", "titre": "Méthode et sources", "court": "Méthode", "svg": _vue_methode(), "legende": "EN 1993-1-8 §4 · méthode ELS du bureau"})
    return out


def _taux(r: Dict) -> Optional[float]:
    return r["taux_max"] if r.get("statut") != ET.REFUS else None


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : son cas le plus chargé."""
    vals = [(t, i) for i, r in enumerate(resultats) if (t := _taux(r)) is not None]
    if not vals:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    t, i = max(vals)
    niv = C.niveau(t)
    g = resultats[i]["gouvernant"]
    nom_ch = _chaine_de(resultats[i]).get(g["cordon"], g["cordon"])
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1],
            "gouvernant": f"{etat['cas'][i].get('nom') or f'Cas {i + 1}'} · {nom_ch}, {g['position']}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """Rien : les cas de la note sont la LIGNE de l'élément, en tête de la colonne de gauche (``arbre``)."""
    return ""


def _ligne_methode(res: Dict) -> Tuple[str, str]:
    """Le pied du résultat : la méthode et son acier (la ligne grise de la carte RÉSULTAT de l'écran Streamlit)."""
    m = res["methode"]
    ac = res["acier"]
    if m.mode == K.ELS:
        return (f"ELS du bureau · k = {C.nb(m.k, 3)}", f"τ<sub>adm</sub>{C.NBSP}={C.NBSP}{C.fr(K.tau_adm_bureau(m.k, m.g), 1)}{C.NBSP}N/mm²")
    return (f"Eurocode {CRITERES[m.ec].split(' (')[0]} · {C.esc(ac['nuance_ref'])}", f"γ<sub>M2</sub>{C.NBSP}={C.NBSP}{C.nb(m.gamma_M2)}")


def _efforts_saisis(cas: Dict) -> List[Tuple[str, float, str]]:
    """Les efforts saisis NON NULS du cas : (symbole du moteur — « V_z » —, valeur, unité), dans l'ordre de l'écran."""
    return [(lib.split(" (")[0], float(cas["efforts"][cle]), unite) for cle, lib, unite, _ in CF.EFFORTS[cas["config"]]
            if cas["efforts"].get(cle)]


def _entete(cas: Dict, res: Dict, v: Dict) -> str:
    """L'en-tête technique d'une vue : son titre ; la configuration ; la méthode et son coefficient ; à droite les efforts
    saisis (non nuls) ; ce qu'on lit sur le dessin quand ce n'est pas la configuration elle-même."""
    cfg = cas["config"]
    lib = f"{cfg} · {CF.CONFIGS[cfg]['libelle']}"
    meth, coef = _ligne_methode(res)
    bande = [(sym.partition("_")[0], sym.partition("_")[2], x, u) for sym, x, u in _efforts_saisis(cas)]
    return E.entete(v["titre"].upper(), C.esc(lib), [f"{_texte(meth)} · {_texte(coef)}"], bande or None,
                    "" if v.get("legende") == lib else v.get("legende", ""))


def resultat(etat: Dict) -> Dict:
    """Le calcul et sa mise en forme : la tête (le nom du cas ; la configuration, la méthode), le verdict, le bloc
    « Cordons » (une ligne par cordon : sa gorge, qui se choisit ; ce qu'il compare à son point le plus chargé ; sa
    jauge), le bloc « Les quatre lectures », les messages du moteur ; les fenêtres de calcul, les vues sous leur en-tête
    technique, le résumé de la note et la colonne de gauche."""
    cas = _cas(etat)
    res = ET.verifier(cas)
    resultats = [res if i == etat["actif"] else ET.verifier(c) for i, c in enumerate(etat["cas"])]
    g = res.get("gouvernant")
    ch_de = _chaine_de(res)
    blocs, volets = "", ""
    msgs = list(res.get("messages", []))
    cfg = cas["config"]
    lecture = "méthode ELS du bureau" if cas["methode"]["mode"] == K.ELS else f"Eurocode {CRITERES[cas['methode']['ec']].split(' (')[0]}"
    tete = E.tete_resultats(C.esc(cas.get("nom") or "Cas"), f"{cfg} · {C.esc(CF.CONFIGS[cfg]['libelle'])} · {lecture}")
    if res["statut"] == ET.REFUS:
        niv = "x"
        # la raison du refus (l'écran Streamlit montre le premier message, souvent une simple note d'espacement)
        raison = next((x for x in msgs if x.startswith("Refus")), msgs[0] if msgs else "données manquantes")
        if raison in msgs:
            msgs.remove(raison)
        verdict = f'<div class="verdict"><div class="gov"><b>Calcul impossible</b>{_texte_indices(texte_fr(raison))}</div></div>'
    else:
        t = res["taux_max"]
        niv = C.niveau(t)
        nom_g = ch_de.get(g["cordon"], g["cordon"])
        i_g = next((i for i, c in enumerate(res["chaines"]) if c["nom"] == ch_de.get(g["cordon"])), 0)
        verdict = (f'<div class="verdict" data-calcul="c{i_g}" title="Voir le calcul">{C.chip(C.pct(t), niv, xl=True)}<div class="gov">'
                   f'<b>{C.esc(_maj(nom_g))} · {C.esc(g["position"])}</b>'
                   f'{"pleine pénétration : " if g.get("pleine") else ""}{_verdict_comparaison(res, g)}</div></div>')
        hyps = _hypotheses(res)
        rows = ""
        optimaux = {}
        for i, ch in enumerate(res["chaines"]):
            info = _param_gorge(cas, ch)
            leger = ""
            if info:
                if info["role"] not in optimaux:
                    full = _info_role(cas, res, info["role"])
                    optimaux[info["role"]] = gorge_optimale(cas, full) if full else None
                o = optimaux[info["role"]]
                if o and abs(o["valeur"] - float(info["valeur"])) > 1e-9:
                    leger = f"La plus petite {'pénétration' if info['sym'].startswith('p') else 'gorge'} qui tient : {C.nb(o['valeur'])} mm · {_pct(o['taux'])}"
            # ce que le cordon compare à SON point le plus chargé (la vérification de la méthode choisie), et où
            pt = _pire_point(res, ch["nom"])
            quoi = f"{_verdict_comparaison(res, pt)} · {C.esc(pt['position'])}" if pt else _longueur(ch)
            rows += C.ligne_verif(_maj(ch["nom"]), quoi, C.pct(ch["taux"]), C.niveau(ch["taux"]), calcul=f"c{i}", pk=info["role"] if info else "",
                                  piece=_piece(ch, info), gov=ch["nom"] == ch_de.get(g["cordon"]), leger=leger)
            volets += _fenetre_chaine(i, ch, res, info, hyps)
        if cfg in GROUPES_PROFIL:
            rows += C.ligne_verif("Profilé soudé" if cfg == "C5" else "Tube soudé", f"sur la platine · t<sub>p</sub> = {C.nb(cas['params'].get('tp'))} mm",
                                  C.pct(t), niv, calcul=f"c{i_g}", pk="profil", piece=C.esc(cas["params"].get("profil", "")))
        nature = "Efforts de service" if res["methode"].mode == K.ELS else "Efforts de calcul (ELU)"
        saisis = " · ".join(f"{C.symb_math(sym)} = {C.nb(x).replace('-', '−')}{C.NBSP}{u}" for sym, x, u in _efforts_saisis(cas)) or "aucun effort"
        blocs = E.bloc(0, "Cordons", CLASSES_BLOC.get(niv, "inc"), f"{C.STATUTS[niv][1]} · {_pct(t)}", f"{nature} : {saisis}", rows, theme=True,
                       calcul=f"c{i_g}")
        comp = comparaison(res)
        if comp:
            rows = "".join(C.ligne_verif(c["libelle"], "méthode choisie ◀" if c["choisie"] else "à titre indicatif", C.pct(c["taux"]),
                                         C.niveau(c["taux"]), calcul="m") for c in comp)
            blocs += E.bloc(1, "Les quatre lectures", "inc", "mêmes efforts", "chaque méthode à son point le plus chargé ; la méthode choisie "
                            "décide du statut", rows, theme=True, calcul="m")
            volets += _fenetre_comparaison(res, hyps)
    messages = "".join(C.message(_texte_indices(texte_fr(msg)), message_grave(msg)) for msg in msgs)
    meth, coef = _ligne_methode(res)
    pied = (f'<div class="pfoot"><div class="mass"><span>{meth}</span><b>{coef}</b></div>'
            f'<div class="legend">{C.legende()}</div></div>')
    vs = vues(res)
    for v in vs:
        v["entete"] = _entete(cas, res, v)
    # le statut de la NOTE (dès deux cas) : le moins bon de ses cas ; seul, le verdict le dit déjà
    nivs = [C.niveau(x) if (x := _taux(r)) is not None else "x" for r in resultats]
    statut = C.statut(next((n for n in ("x", "r", "f") if n in nivs), "p")) if len(resultats) > 1 else ""
    corps = f'{tete}{verdict}<div class="et-blocs">{blocs}</div>{messages}' if blocs else f"{tete}{verdict}{messages}"
    return {"resultat": f'<div class="pbody">{corps}</div>{pied}', "volets": volets, "vues": vs, "phrase": "", "statut_html": statut,
            "resume": resume_note(etat, resultats), "donnees": arbre(etat, res), "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ la note PDF
# le générateur VALIDÉ de l'écran Streamlit (acier/soudures/note.py), deux pages par cas (api_web/note_cas.py) ; sa
# page 2 reçoit l'échelle et le nombre de points montrés en page 1 ; l'identification imprimée est celle de la note du
# site (celle de l'état, « identification », venue de Streamlit, ne s'imprime pas et ne se montre plus)
NOTE = NC.Generateur("acier.soudures.note", "Soudures", lambda n1: (n1.echelle, n1.points_montres),
                     ("acier/soudures/note.py", "acier/assemblages/poutre_poutre/doubles_cornieres/rapport.py"))


def telecharger(etat: Dict, quoi: str, infos: Optional[Dict] = None) -> Optional[Dict]:
    """« note_pdf » : la note de calcul PDF de TOUTE la note (chaque cas, deux pages), faite par le générateur validé de
    l'écran Streamlit ; ``infos`` : le nom, le chantier, l'indice et la date de la note du site."""
    if quoi != "note_pdf":
        return None
    return NC.fichier(NOTE, etat, infos)


# ------------------------------------------------------------------ les sélecteurs : la gorge, le profilé
def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """La ligne choisie dans un sélecteur : une gorge (« a = 6 mm ») ou un profilé du catalogue."""
    cas = _cas(etat)
    if role == "profil":
        msg = modifier(etat, ["params", "profil"], nom)
        if msg:
            raise ValueError(msg)
        return
    res = ET.verifier(cas)
    info = _info_role(cas, res, role)
    # la ligne du sélecteur se nomme « a = 6 mm » (``_nom_gorge``) ; un nombre seul est aussi lu
    m = re.fullmatch(r"(?:[A-Za-z]\w*\s*=\s*)?(-?\d+(?:[.,]\d+)?)(?:\s*mm)?", str(nom).strip())
    if info is None or m is None:
        raise ValueError(f"choix inconnu : {role} = {nom}")
    msg = modifier(etat, info["chemin"], m.group(1))
    if msg:
        raise ValueError(msg)


def _nom_gorge(info: Dict, v: float) -> str:
    return f"{info['sym'].split('_')[0]} = {C.nb(v)} mm"


def selecteur(etat: Dict, role: str, familles: Optional[Sequence[str]] = None, froid: bool = True) -> str:
    """Le sélecteur d'une gorge (chaque gorge recalculée, la plus petite qui tient ★) ou du profilé soudé (C5, C6 :
    chaque profilé de la famille recalculé, le plus léger qui tient ★), triés par ordre de grandeur."""
    cas = _cas(etat)
    if role == "profil":
        return _selecteur_profil(cas, familles)
    res = ET.verifier(cas)
    info = _info_role(cas, res, role)
    if info is None:
        raise ValueError(f"sélecteur inconnu : {role}")
    cands = candidats_gorge(cas, info)
    opt = next((c for c in cands if c["tient"]), None)
    rows = []
    for c in cands:
        if c["taux"] is None:
            continue
        g = c["gouvernant"] or {}
        rows.append({"nom": _nom_gorge(info, c["valeur"]), "col": C.fr(c["A_w"] or 0, 0), "taux": C.pct(min(c["taux"], 9.99)), "niveau": C.niveau(c["taux"]),
                     "cur": abs(c["valeur"] - float(info["valeur"])) < 1e-9, "opt": opt is not None and c is opt,
                     "prop": (f"{_nom_gorge(info, c['valeur'])} · A<sub>w</sub> = {C.fr(c['A_w'] or 0, 0)} mm² · η = {_pct(c['taux'])}"
                              + (" · un cordon ne compte pas (gorge &lt; 3 mm ou trop court)" if c["exclus"] else "")
                              + (f" · {C.esc(_chaine_de_point(cas, info, c['valeur'], g))}" if g else ""))})
    noms = [r["nom"] for r in rows]
    i_c = next((i for i, r in enumerate(rows) if r["cur"]), 0)
    i_o = next((i for i, r in enumerate(rows) if r["opt"]), i_c)
    quoi = "Pénétration" if info["sym"].startswith("p") else "Gorge"
    cibles = ", ".join(info["chaines"]) or info.get("el", "")
    del noms
    return C.selecteur(role, f"{quoi} · {C.esc(cibles)}", "triées par épaisseur · chaque ligne recalculée · un clic la retient", "", rows,
                       max(0, min(i_o, i_c) - 4), "mm² de gorges", etoile=f"la plus petite {quoi.lower()} qui tient")


def _chaine_de_point(cas: Dict, info: Dict, v: float, g: Dict) -> str:
    """Le cordon et le point gouvernants d'une ligne du sélecteur de gorge."""
    return f"gouvernant : {g.get('cordon', '')}, {g.get('position', '')}"


def _selecteur_profil(cas: Dict, familles: Optional[Sequence[str]]) -> str:
    cfg = cas["config"]
    if cfg not in GROUPES_PROFIL:
        raise ValueError("pas de profilé dans cette configuration")
    groupe = GROUPES_PROFIL[cfg]
    cur = cas["params"].get("profil")
    rec = CAT.par_nom().get(cur)
    fams = [f for f in (familles or []) if f in _familles(groupe)] or ([rec["famille"]] if rec else _familles(groupe)[:1])
    cands = []
    for nom, r in CAT.par_nom().items():
        if r["famille"] not in fams:
            continue
        c = copy.deepcopy(cas)
        c["params"]["profil"] = nom
        rr = ET.verifier(c)
        if rr["statut"] == ET.REFUS:
            continue
        cands.append((r["masse"], nom, rr["taux_max"], r))
    cands.sort(key=lambda x: (x[0], x[1]))
    best = next((x for x in cands if x[2] <= 1.0), None)

    def prop(r):
        dims = (f"d × t = {C.nb(r['d'])} × {C.nb(r['t'])} mm" if r.get("d") else
                f"h × b = {C.nb(r['h'])} × {C.nb(r['b'])} mm · " + (f"t = {C.nb(r['t'])} mm" if r.get("t") else
                                                                f"t<sub>w</sub> = {C.nb(r['tw'])} · t<sub>f</sub> = {C.nb(r['tf'])} mm"))
        return f"{C.esc(r['nom'])} · {dims} · {C.fr(r['masse'], 1)} kg/m · {C.esc(FA.FAMILLES[r['famille']].get('norme', ''))}"
    rows = [{"nom": nom, "col": C.fr(masse, 1), "taux": C.pct(min(t, 9.99)), "niveau": C.niveau(t), "cur": nom == cur,
             "opt": best is not None and nom == best[1], "prop": prop(r)} for masse, nom, t, r in cands]
    noms = [x[1] for x in cands]
    i_c = noms.index(cur) if cur in noms else 0
    i_o = noms.index(best[1]) if best else i_c
    filtres = "".join(f'<button type="button" class="fchip{" on" if f in fams else ""}" data-famille="{C.esc(f)}">{C.esc(f)}</button>'
                      for f in _familles(groupe))
    return C.selecteur("profil", "Profilé soudé · même soudure" if cfg == "C5" else "Tube soudé · même soudure",
                       "triés par masse · chaque profilé recalculé avec les mêmes cordons · un clic le retient", filtres, rows,
                       max(0, min(i_o, i_c) - 4), "kg/m")


# ------------------------------------------------------------------ le parcours : ce que le parcours commun n'atteint pas seul
def _etat_demo(cfg: str, **params) -> Dict:
    e = nouvel_etat()
    ET.changer_config(e["cas"][0], cfg)
    e["cas"][0]["params"].update(params)
    return normaliser(e)


PARCOURS = [
    {"action": "selecteur", "role": "profil", "etat": _etat_demo("C5")},
    {"action": "choisir", "role": "profil", "nom": "IPE 300", "etat": _etat_demo("C5")},
    {"action": "selecteur", "role": "profil", "familles": ["CHS", "RHS"], "etat": _etat_demo("C6")},
    {"action": "selecteur", "role": "gorge-a_w", "etat": _etat_demo("C5")},
    {"action": "selecteur", "role": "gorge-el1", "etat": _etat_demo("C8")},
    {"action": "choisir", "role": "gorge-el0", "nom": "a = 8 mm", "etat": _etat_demo("C8")},
    {"action": "modifier", "chemin": ["params", "elements", "+"], "valeur": "arc", "etat": _etat_demo("C8")},
    {"action": "modifier", "chemin": ["params", "elements", "0", "type"], "valeur": "arc", "etat": _etat_demo("C8")},
    {"action": "modifier", "chemin": ["params", "elements", "1", "-"], "valeur": 1, "etat": _etat_demo("C8")},
    {"action": "modifier", "chemin": ["params", "jonction"], "valeur": "partielle", "etat": _etat_demo("C6", troncons={"n": 4, "l": 40.0})},
    {"action": "modifier", "chemin": ["params", "jonction_semelles"], "valeur": "pleine", "etat": _etat_demo("C5")},
    {"action": "modifier", "chemin": ["params", "penetration"], "valeur": "partielle", "etat": _etat_demo("C7", p=6.0)},
    {"action": "modifier", "chemin": ["methode", "mode"], "valeur": K.ELS, "etat": _etat_demo("C2", jonction="pleine")},
    {"action": "modifier", "chemin": ["params", "poteau_famille"], "valeur": "HEA", "etat": _etat_demo("C4")},
    {"action": "modifier", "chemin": ["params", "profil_famille"], "valeur": "RHS-F", "etat": _etat_demo("C6")},
    {"action": "modifier", "chemin": ["import"], "valeur": {"nom": "soudures_cas.json", "texte": ET.vers_json(_etat_demo("C1"))}},
]

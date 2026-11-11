# -*- coding: utf-8 -*-
"""Le module « Fondations de grues » du site (docs/beton/FONDATIONS_GRUE.md ; l'étude des quatre classeurs :
docs/beton/FONDATIONS_GRUE_ETUDE.md ; le moteur : ``beton/fondation_grue``), sur l'ÉTABLI COMPACT — le format de la
poutre béton armé (api_web/etabli_compact.py, js/compact.js) :

- à gauche, la ligne de la fondation ouverte (son nom, la liste des fondations de la note, copier, supprimer) puis
  ses six rubriques : 1 Fondation (le schéma — 8 pieux sur semelles ou contre une paroi de pieux sécants, 4 pieux en
  cadre, 8 pieux sous poutres primaires —, le chantier, la grue, « reprendre un classeur »), 2 Matériaux,
  3 Géométrie, 4 Tolérances d'exécution, 5 Charges de la grue, 6 Armatures choisies ;
- au centre, la VUE EN PLAN (redessinée à chaque saisie) puis, pour chaque poutre et chaque semelle, son schéma sur
  deux appuis — une rangée par position de tolérance — et ses diagrammes M et V : une courbe par position, l'enveloppe
  en trait fort, les valeurs que le classeur retient en tireté (beton/fondation_grue/schemas.py) ;
- à droite, la tête, le VERDICT (le taux le plus fort et l'état du classeur), un bloc par partie du classeur (poutres,
  semelles, pieux, renfort au droit du pieu, effort horizontal) — chaque ligne dit ce qu'elle compare, l'état que le
  CLASSEUR lui donne (OK, LIMITE, PAS OK, A VERIFIER) et ouvre SA fenêtre de calcul —, les contrôles de géométrie, les
  ÉCARTS relevés dans le classeur (jamais corrigés en silence : les deux chiffres), puis l'export ;
- les FENÊTRES DE CALCUL : chaque formule du classeur avec ses valeurs et la CELLULE d'où elle vient ; les grands
  tableaux des cas de la grue ; le contrôle des données avant export.

Ce fichier MET EN FORME : il ne calcule rien et n'écrit aucune formule ; il choisit parmi ce que le moteur a nommé.
Toute saisie passe par ``moteur.normaliser`` : hors domaine, elle est REFUSÉE avec le message du moteur et la
fondation reste intacte. L'export (RBAR, DXF, tableau) vient de ``beton/fondation_grue/export.py`` : il ne recalcule
rien et refuse de partir quand une vérification du classeur est « PAS OK ».
"""
import copy
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from beton.fondation_grue import classeur as K
from beton.fondation_grue import donnees as D
from beton.fondation_grue import ecarts as EC
from beton.fondation_grue import ecriture as W
from beton.fondation_grue import entree as EN
from beton.fondation_grue import export as X
from beton.fondation_grue import moteur as MO
from beton.fondation_grue import schemas as SC

from . import commun as C
from . import etabli_compact as E

MODULE = "fondation_grue"
PAGE = "fondations-grues"
VERSION = 1                     # l'état de la note : {"version": 1, "fondations": [...], "actif": 0}
GENRE = E.Genre("Fondations", "la fondation", "cette fondation", "une fondation")
NOM_DEFAUT = "F1"               # « F1 », « F2 »… : court, comme « V1 » de la poutre (la ligne de l'élément, au doigt, n'a pas la place d'un nom long)

S8, C4, D8 = EN.SEMELLES8, EN.CADRE4, EN.DIAGONALES8
SEGMENTS_COTE_B = ((EN.COTE_SEMELLES, "Semelles"), (EN.COTE_PAROI, "Paroi de pieux sécants"))
SEGMENTS_ETAT_CAS = ((EN.SERVICE, "En service"), (EN.HORS, "Hors service"))
OPTIONS_DIAMETRE = tuple((d, str(d)) for d in EN.DIAMETRES)
# les champs ENTIERS (un nombre de barres, de brins) et les champs à MENU (un diamètre) : par la fin de leur chemin
FINS_ENTIER = ("n", "brins", "brins_base")
FINS_DIAMETRE = ("d",)
TEXTES_CAS = ("nom",)
# le nom COURT d'une vérification (1ʳᵉ ligne d'un résultat) : la fin de sa clé
COURTS = {"hauteur": "Hauteur utile", "inf": "Armatures inf.", "sup": "Armatures sup.",
          "etriers": "Étriers", "etriers_extremites": "Étriers extrémités", "etriers_centre": "Étriers centre",
          "etriers_appui": "Étriers appui", "etriers_travee": "Étriers travée", "pieux": "Pieu le plus chargé",
          "perfos": "Perfos"}
# le nom COURT d'un bloc (son en-tête dans la colonne des résultats) : par sa clé ; le nom du classeur reste dans les fenêtres
COURTS_BLOC = {"support": "Poutre support", "semelle": "Semelle sur pieux", "pieux": "Pieux", "renfort": "Renfort au pieu",
               "horizontal": "Effort horizontal", "poutre": "Poutre", "secondaire": "Poutre secondaire", "primaire": "Poutre primaire"}
# ce qu'une vérification compare : (la grandeur calculée, la grandeur disponible), par la fin de sa clé
COMPARE = {"hauteur": ("néc.", "d"), "pieux": ("sollicitation", "capacité")}
COMPARE_ETRIERS = ("pas", "pas max")
COMPARE_DEFAUT = ("néc.", "posé")
NIVEAUX_ETAT = {MO.OK: "p", MO.WARN: "v", MO.NOK: "r", MO.INFO: "c"}
CLASSES_BLOC = {MO.OK: "ok", MO.WARN: "warn", MO.NOK: "nok"}
MOTS_BLOC = {MO.OK: "Conforme", MO.WARN: "À vérifier", MO.NOK: "Non conforme"}
# la couleur d'un ÉCART relevé dans le classeur : orange quand il va contre la sécurité, qu'il manque une vérification ou
# que deux cellules se contredisent ; neutre quand il est du côté de la sécurité ou qu'il borne le domaine du classeur
NIVEAUX_GRAVITE = {EC.NON_CONSERVATIF: "v", EC.INCOHERENCE: "v", EC.MANQUE: "v", EC.CONSERVATIF: "c", EC.LIMITE: "c"}
DECIMALES = {"mm": 0, "mm²": 0, "cm": 1, "m": 2, "kN": 1, "kN/m": 2, "kN·m": 1, "T": 1, "N/mm²": 3, "%": 1}

# les requêtes de plus pour le parcours (api_web/parcours.py) et le relevé du paquet : chaque schéma, la paroi de pieux
# sécants, les quatre classeurs, un cas de la grue ajouté puis retiré, des saisies refusées
PARCOURS = [
    {"action": "modifier", "chemin": ["type"], "valeur": C4},
    {"action": "modifier", "chemin": ["type"], "valeur": D8},
    {"action": "modifier", "chemin": ["cote_b"], "valeur": EN.COTE_PAROI},
    {"action": "modifier", "chemin": ["exemple"], "valeur": "GWW"},
    {"action": "modifier", "chemin": ["exemple"], "valeur": "VAR4"},
    {"action": "modifier", "chemin": ["exemple"], "valeur": "VAR8"},
    {"action": "modifier", "chemin": ["charges", "cas", "+"], "valeur": "1"},
    {"action": "modifier", "chemin": ["charges", "cas", "0", "-"], "valeur": "1"},
    {"action": "modifier", "chemin": ["e_support"], "valeur": "80"},
    {"action": "modifier", "chemin": ["support", "inf", "0", "n"], "valeur": "2"},
    {"action": "modifier", "chemin": ["pieux", "capacite"], "valeur": "40"},
    {"action": "telecharger", "quoi": "rbar"},
    {"action": "telecharger", "quoi": "dxf"},
    {"action": "telecharger", "quoi": "tableau"},
]


# ------------------------------------------------------------------ l'état de la note : des fondations
def nouvel_etat() -> Dict:
    """Une note neuve : une fondation à 8 pieux sur quatre semelles, aux valeurs du classeur BASE (chantier et grue
    vides)."""
    return {"version": VERSION, "fondations": [MO.nouvelle_fondation(S8, NOM_DEFAUT)], "actif": 0}


def _indice(x) -> int:
    """L'indice de la fondation ouverte, lu sans faire tomber la note : un nombre entier fini, sinon la première."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return 0
    return int(x)


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu par le moteur ; une fondation illisible lève une erreur
    claire. Une note sans fondation est une note neuve ; une note d'une version plus récente est refusée."""
    fondations = etat.get("fondations") if isinstance(etat, dict) else None
    if not fondations:
        return nouvel_etat()
    if not isinstance(fondations, list):
        raise ValueError("note abîmée : « fondations » doit être une liste")
    version = etat.get("version", VERSION)
    if isinstance(version, (int, float)) and not isinstance(version, bool) and version > VERSION:
        raise ValueError(f"note enregistrée par une version plus récente du module (version {version}) : elle ne se relit pas ici")
    out = {"version": VERSION, "fondations": []}
    for i, f in enumerate(fondations):
        if not isinstance(f, dict):
            raise ValueError(f"fondation n° {i + 1} illisible : un groupe de champs est attendu")
        try:
            out["fondations"].append(MO.normaliser(f))
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"fondation « {f.get('nom', i + 1)} » : {e}") from None
    out["actif"] = min(max(0, _indice(etat.get("actif"))), len(out["fondations"]) - 1)
    return out


def _fondation(etat: Dict) -> Dict:
    return etat["fondations"][etat["actif"]]


def _nom_libre(fs: List[Dict]) -> str:
    pris = {f.get("nom") for f in fs}
    k = len(fs) + 1
    while f"F{k}" in pris:
        k += 1
    return f"F{k}"


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les fondations de la note : en ouvrir une, en ajouter une (du schéma de celle qui est ouverte), en copier une
    (la copie s'insère juste après), en supprimer une (jamais la dernière)."""
    fs = etat["fondations"]
    if action == "activer":
        etat["actif"] = min(max(0, _indice(index)), len(fs) - 1)
    elif action == "ajouter":
        fs.append(MO.nouvelle_fondation(_fondation(etat)["type"], _nom_libre(fs)))
        etat["actif"] = len(fs) - 1
    elif action == "dupliquer":
        i = etat["actif"] if index is None else min(max(0, _indice(index)), len(fs) - 1)
        f = copy.deepcopy(fs[i])
        f["nom"] = f"{f.get('nom') or 'Fondation'} (copie)"[:EN.NOM_MAX]
        fs.insert(i + 1, f)
        etat["actif"] = i + 1
    elif action == "supprimer" and len(fs) > 1:
        i = min(max(0, _indice(index)), len(fs) - 1)
        del fs[i]
        if etat["actif"] > i or etat["actif"] >= len(fs):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
def _changer_schema(f: Dict, type_: str) -> Dict:
    """Le schéma change : la fondation repart des valeurs du classeur de ce schéma ; le nom, le chantier, la grue et
    les matériaux sont gardés."""
    neuve = MO.nouvelle_fondation(type_, f.get("nom") or NOM_DEFAUT)
    for cle in ("projet", "grue", "beton", "acier"):
        neuve[cle] = f[cle]
    return neuve


def _cas_neuf(cas: List[Dict]) -> Dict:
    pris = {c["nom"] for c in cas}
    k = len(cas) + 1
    while f"CAS {k}" in pris:
        k += 1
    return {"nom": f"CAS {k}", "etat": EN.SERVICE, "A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la fondation ouverte ; ``chemin`` : « chassis », « support/inf/0/n », « charges/cas/2/B »…
    Le moteur relit la fondation : une valeur hors de son domaine est REFUSÉE avec son message, et la fondation reste
    intacte. Les saisies qui font plus qu'écrire une valeur : « type » (le schéma : la fondation repart des valeurs de
    son classeur), « exemple » (toutes les valeurs d'un des quatre classeurs ; le nom est gardé), « charges/cas/+ » et
    « charges/cas/<i>/- » (un cas de la grue de plus, de moins)."""
    f = _fondation(etat)
    avant = copy.deepcopy(f)
    cles = [str(c) for c in chemin]
    cle = "/".join(cles)
    fin = cles[-1] if cles else ""
    try:
        if cle == "nom":
            nom = str(valeur if valeur is not None else "").strip()
            if len(nom) > EN.NOM_MAX:
                return f"Nom trop long : {len(nom)} caractères (au plus {EN.NOM_MAX})"
            f["nom"] = nom or NOM_DEFAUT
        elif cle == "type":
            if valeur not in EN.TYPES:
                return f"Schéma inconnu : « {valeur} »"
            if valeur != f["type"]:
                f = _changer_schema(f, valeur)
        elif cle == "exemple":
            if valeur in ("", None):
                return None
            if valeur not in EN.EXEMPLES:
                return f"Classeur inconnu : « {valeur} »"
            nom = f.get("nom") or NOM_DEFAUT
            f = MO.exemple(valeur)
            f["nom"] = nom
        elif cle == "charges/cas/+":
            if f["type"] != S8:
                return "Ce schéma n'a pas de tableau des cas de la grue"
            if len(f["charges"]["cas"]) >= EN.N_CAS_MAX:
                return f"Au plus {EN.N_CAS_MAX} cas de la grue"
            f["charges"]["cas"].append(_cas_neuf(f["charges"]["cas"]))
        elif len(cles) == 4 and cles[:2] == ["charges", "cas"] and fin == "-":
            cas = f["charges"]["cas"] if f["type"] == S8 else []
            i = int(cles[2]) if cles[2].isdigit() else -1
            if not 0 <= i < len(cas):
                return "Cas de la grue introuvable"
            if len(cas) == 1:
                return "Il faut au moins un cas de la grue"
            del cas[i]
        elif cle in EN.CHAMPS_TEXTE or (cles[:2] == ["charges", "cas"] and fin in TEXTES_CAS):
            EN.poser(f, cle, str(valeur if valeur is not None else ""))
        elif cle in ("beton", "acier", "cote_b") or (cles[:2] == ["charges", "cas"] and fin == "etat"):
            EN.poser(f, cle, valeur)
        else:
            EN.lire(f, cle)                                     # KeyError : le champ n'existe pas dans ce schéma
            x = C.lire_nombre(valeur)
            if x is None:
                return f"Nombre illisible : « {valeur} »"
            if fin in FINS_ENTIER or fin in FINS_DIAMETRE:
                if not float(x).is_integer():
                    return f"Nombre entier attendu : « {valeur} »"
                x = int(x)
            EN.poser(f, cle, x)
        etat["fondations"][etat["actif"]] = MO.normaliser(f)
    except KeyError:
        etat["fondations"][etat["actif"]] = avant
        return f"Champ inconnu : « {cle} »"
    except (ValueError, TypeError, IndexError) as e:
        etat["fondations"][etat["actif"]] = avant
        texte = str(e)
        return f"Saisie refusée : {texte[:1].lower() + texte[1:]}"
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Aucun sélecteur dans ce module (les armatures se choisissent dans la rubrique 6)."""
    return None


def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    return ""


# ------------------------------------------------------------------ les nombres
def _n(x: Optional[float], unite: str = "") -> str:
    """Un résultat à la précision d'usage de son unité (mm et mm² entiers, kN et kN·m au dixième, τ au millième)."""
    if x is None:
        return "—"
    return C.fr(x, DECIMALES.get(unite, 2))


def _nu(x: Optional[float], unite: str) -> str:
    """Un résultat suivi de son unité (insécable)."""
    if x is None:
        return "—"
    return f"{_n(x, unite)}{C.NBSP}{unite}" if unite else _n(x, unite)


def _g(x: float) -> str:
    """Une donnée saisie, écrite sans zéros inutiles (« 140 », « 62,5 »)."""
    return C.nb(x, 3)


NOMS_VARIANTE = {"diagonale": "en diagonale", "cote": "par côté"}


def _val_ecart(x) -> str:
    """Une valeur d'un écart (« classeur », « redémontré ») : un nombre, ou plusieurs nommés (« en diagonale 27,05 ·
    par côté 33,56 ») quand la grandeur se redémontre de deux façons ; « — » sans valeur."""
    if x is None:
        return "—"
    if isinstance(x, dict):
        return " · ".join(f"{C.esc(NOMS_VARIANTE.get(k, k))} {_g(round(v, 2))}" for k, v in x.items())
    return _g(round(x, 2))


def _pct(t: Optional[float]) -> str:
    return f"{C.pc(C.pct(t, 1))}{C.NBSP}%" if t is not None else "—"


def _niv(v: Dict) -> str:
    """La couleur d'une vérification : celle de l'ÉTAT que le classeur lui donne — rouge « PAS OK », orange « LIMITE »
    ou « A VERIFIER » ; « OK » prend la couleur de son taux."""
    if v["etat"] != MO.OK:
        return NIVEAUX_ETAT[v["etat"]]
    return C.niveau(v["taux"]) if v["taux"] is not None else "p"


def _niv_res(r: Dict) -> str:
    if r["statut"] != MO.OK:
        return NIVEAUX_ETAT[r["statut"]]
    return C.niveau(r["taux_max"])


# ------------------------------------------------------------------ le cadre Données : les six rubriques
def _num(f: Dict, chemin: str, unite: str = "", entier: bool = False) -> str:
    return C.num(chemin.split("/"), EN.lire(f, chemin), unite, entier=entier)


def _texte(f: Dict, chemin: str, indice: str = "") -> str:
    return C.texte(chemin.split("/"), EN.lire(f, chemin), indice)


def _diametre(f: Dict, chemin: str) -> str:
    return C.choix(chemin.split("/"), EN.lire(f, chemin), OPTIONS_DIAMETRE)


def _lit(f: Dict, base: str, nom: str, aide: str = "") -> str:
    """Un groupe de barres « n Ø d » sur une ligne : le nombre (entier) et le diamètre (menu)."""
    return C.ligne2((nom, _num(f, base + "/n", "", True), aide), ("Ø", _diametre(f, base + "/d"), ""))


def _cadres(f: Dict, base: str, nom: str, aide: str = "") -> str:
    """Des étriers : les brins et le diamètre sur une ligne, le pas sur la suivante."""
    return (C.ligne2((nom, _num(f, base + "/brins", "", True), aide or "Nombre de brins verticaux de l'étrier (2 brins par cadre)."),
                     ("Ø", _diametre(f, base + "/d"), ""))
            + C.ligne("pas", _num(f, base + "/pas", "mm"), "s"))


def _titre(texte: str) -> str:
    return f'<div class="fg-st">{C.esc(texte)}</div>'


def _rubrique_fondation(f: Dict) -> str:
    """1 Fondation : le schéma, le côté b (semelles ou paroi de pieux sécants), le chantier, la grue ; et « reprendre
    un classeur » : toutes les valeurs d'un des quatre classeurs validés, pour comparer."""
    s = C.ligne("Schéma", C.choix(["type"], f["type"], [(t, EN.LIBELLES_TYPE[t]) for t in EN.TYPES]), "",
                "Les trois schémas des classeurs du bureau. Changer de schéma repart des valeurs de son classeur ; le nom, le chantier, "
                "la grue et les matériaux sont gardés.", full=True)
    if f["type"] == S8:
        s += C.ligne("Côté b", C.segments(["cote_b"], f["cote_b"], SEGMENTS_COTE_B), "",
                     "Du côté b, les poutres support reposent sur deux semelles de deux pieux (8 pieux) ou directement sur une paroi de "
                     "pieux sécants (4 pieux et la paroi). Le calcul des poutres est le même : seule la distance b compte.", full=True)
    s += C.ligne("Chantier", _texte(f, "projet", "nom du chantier"), "", "", full=True)
    s += C.ligne("Grue", _texte(f, "grue", "type de grue"), "", "", full=True)
    if f["type"] == S8:
        s += C.ligne2(("Flèche", _texte(f, "fleche"), ""), ("H crochet", _texte(f, "hauteur_crochet"), "Hauteur sous crochet."))
        s += C.ligne2(("Niveau", _texte(f, "niveau_beton"), "Niveau du béton de la fondation."),
                      ("Châssis", _texte(f, "dimensions_chassis"), "Les dimensions du châssis, telles que la fiche de la grue les donne (texte)."))
    s += C.ligne("Reprendre", C.choix(["exemple"], "", [("", "— un classeur validé —")] + [(k, EN.LIBELLES_EXEMPLE[k]) for k in EN.EXEMPLES]), "",
                 "Reprend TOUTES les valeurs d'un des quatre classeurs validés par le bureau (le nom de la fondation est gardé) : pour "
                 "comparer le module au classeur, ligne par ligne.", full=True)
    return s


def _rubrique_materiaux(f: Dict, r: Dict) -> str:
    m = r["materiaux"]
    s = C.ligne2(("Béton", C.choix(["beton"], f["beton"], [(b, b) for b in D.BETONS]), "La classe du béton fixe R'wk, donc σ'b = R'wk / 2,7 et les "
                  "seuils de cisaillement (base de données du classeur)."),
                 ("Acier", C.choix(["acier"], f["acier"], [(a, a) for a in D.ACIERS]), "La nuance fixe σ_adm = f_y / 1,5 et le coefficient μ."))
    if f["type"] == S8:
        s += C.ligne("enrobage", _num(f, "enrobage", "mm"), "c", "Enrobage des armatures : il entre dans la hauteur utile d (h − c − Ø étrier − Ø/2).")
    s += C.aide(f"R'<sub>wk</sub> = {_g(m['Rwk'])} · σ'<sub>b</sub> = {C.fr(m['sigma_b'], 2)} · σ<sub>adm</sub> = {C.fr(m['sigma_adm'], 2)} N/mm² · "
                f"μ = {C.fr(m['mu'], 4)}")
    return s


def _rubrique_geometrie(f: Dict) -> str:
    t = f["type"]
    s = C.ligne("châssis", _num(f, "chassis", "cm"), "L<sub>c</sub>", "La largeur du châssis de la grue : l'entraxe de ses pieds (carré).")
    if t == S8:
        s += C.ligne2(("a", _num(f, "dist_a", "cm"), "Distance du pied de la grue à l'axe de la semelle, du côté a."),
                      ("b", _num(f, "dist_b", "cm"), "Distance du pied de la grue à l'axe de l'appui du côté b (semelle ou paroi)."))
        s += _titre("Poutre support châssis")
        s += C.ligne2(("b", _num(f, "support/b", "cm"), "Largeur."), ("h", _num(f, "support/h", "cm"), "Hauteur."))
        s += _titre("Semelle sur pieux")
        s += C.ligne2(("b", _num(f, "semelle/b", "cm"), "Largeur."), ("h", _num(f, "semelle/h", "cm"), "Hauteur."))
        s += C.ligne("longueur", _num(f, "semelle/L", "cm"), "L<sub>s</sub>")
        s += C.ligne("appui", _num(f, "c_appui", "cm"), "c", "Longueur sur laquelle la poutre support appuie sur la semelle (charge répartie sur c).")
        s += _titre("Pieux")
        s += C.ligne2(("Ø", _num(f, "pieux/D", "cm"), "Diamètre des pieux."), ("entraxe", _num(f, "pieux/entraxe", "cm"), "Entredistance des deux pieux d'une semelle."))
        s += C.ligne("capacité", _num(f, "pieux/capacite", "T"), "", "Capacité portante d'un pieu, en tonnes (à l'ELS), donnée par l'étude de sol.")
    elif t == C4:
        s += _titre("Poutre")
        s += C.ligne2(("b", _num(f, "poutre/b", "cm"), "Largeur."), ("h", _num(f, "poutre/h", "cm"), "Hauteur."))
        s += _titre("Pieux")
        s += C.ligne2(("Ø", _num(f, "pieux/D", "cm"), "Diamètre des pieux."), ("capacité", _num(f, "pieux/capacite", "kN"), "Capacité portante d'un pieu (kN)."))
    else:
        s += _titre("Poutre secondaire")
        s += C.ligne2(("b", _num(f, "secondaire/b", "cm"), "Largeur."), ("h", _num(f, "secondaire/h", "cm"), "Hauteur."))
        s += _titre("Poutre primaire (à 45°, sur 2 pieux)")
        s += C.ligne2(("b", _num(f, "primaire/b", "cm"), "Largeur."), ("h", _num(f, "primaire/h", "cm"), "Hauteur."))
        s += C.ligne("longueur", _num(f, "primaire/L", "cm"), "L<sub>p</sub>")
        s += _titre("Pieux")
        s += C.ligne2(("Ø", _num(f, "pieux/D", "cm"), "Diamètre des pieux."), ("entraxe", _num(f, "pieux/entraxe", "cm"), "Entredistance des deux pieux d'une poutre primaire."))
    return s


AIDE_TOLERANCE = ("Tolérance d'exécution : le pieu (ou le châssis) peut être décalé de e. Le module calcule la poutre sur deux appuis dans "
                  "chaque position décalée et retient l'enveloppe.")


def _rubrique_tolerances(f: Dict) -> str:
    t = f["type"]
    if t == S8:
        return (C.ligne("poutre support", _num(f, "e_support", "cm"), "e", AIDE_TOLERANCE)
                + C.ligne("semelle", _num(f, "e_semelle", "cm"), "e", AIDE_TOLERANCE))
    if t == C4:
        return C.ligne("pieux", _num(f, "e", "cm"), "e", AIDE_TOLERANCE)
    return (C.ligne("poutre secondaire", _num(f, "e_secondaire", "cm"), "e", AIDE_TOLERANCE)
            + C.ligne("poutre primaire", _num(f, "e_primaire", "cm"), "e", AIDE_TOLERANCE))


def _cas(f: Dict) -> str:
    """Le tableau des cas de la grue (8 pieux sur semelles) : un cas par carte — son nom, en service ou hors service,
    la charge verticale de chaque pied (kN)."""
    cas = f["charges"]["cas"]
    out = ""
    for i, c in enumerate(cas):
        base = ["charges", "cas", i]
        pieds = "".join(f'<label><span>{p}</span></label>{C.num(base + [p], c[p], "kN")}' for p in EN.PIEDS)
        # (retirer : une icône sur la ligne du nom — le chemin finit par « - », comme etabli_compact.retirer)
        retirer = (f'<button type="button" class="et-ret fg-ret" data-chemin="{C._attr_chemin(base + ["-"])}" data-valeur="1" title="Retirer ce cas" '
                   f'aria-label="Retirer ce cas">{C.icon("trash-2")}</button>') if len(cas) > 1 else ""
        out += (f'<div class="fg-cas" data-cas="{i}"><div class="fg-cas-h">{C.texte(base + ["nom"], c["nom"], "nom du cas")}'
                f'{C.segments(base + ["etat"], c["etat"], SEGMENTS_ETAT_CAS)}{retirer}</div><div class="fg-cas-v">{pieds}</div></div>')
    if len(cas) < EN.N_CAS_MAX:
        out += E.plus(["charges", "cas", "+"], "1", "Cas de la grue")
    return out


def _rubrique_charges(f: Dict) -> str:
    ch = f["charges"]
    s = ""
    if f["type"] == S8:
        s += C.aide("Réactions de la grue, par pied (A, B, C, D en tournant), en kN — telles que la fiche de la grue les donne, cas par cas. "
                    "Le module essaie chaque cas dans les huit orientations de la grue.")
        s += _cas(f)
    else:
        s += C.ligne("par pied", _num(f, "charges/P", "kN"), "P", "Réaction verticale maximale d'un pied de la grue.")
    s += _titre("Efforts horizontaux et moment")
    s += C.ligne2(("H<sub>s</sub>", _num(f, "charges/H_s", "kN"), "Charge horizontale en service."),
                  ("H<sub>hs</sub>", _num(f, "charges/H_hs", "kN"), "Charge horizontale hors service."))
    s += C.ligne("moment", _num(f, "charges/M_s", "kN·m"), "M<sub>s</sub>", "Moment en service (torsion du châssis), repris en couple par les pieds.")
    return s


AIDE_PART_V = "Part de l'effort tranchant comptée dans le calcul des étriers de cette zone (le classeur la saisit en %)."


def _section_armatures(f: Dict, base: str, deux: bool, zones: Sequence[Tuple[str, str]], part_v: bool) -> str:
    """Les armatures d'une poutre ou d'une semelle : inférieures, supérieures (deux groupes de barres dans le classeur
    BASE), puis ses zones d'étriers."""
    s = ""
    for cle, nom in (("inf", "inf."), ("sup", "sup.")):
        if deux:
            s += _lit(f, f"{base}/{cle}/0", nom, "Premier groupe de barres.") + _lit(f, f"{base}/{cle}/1", "+", "Second groupe de barres (0 : aucun).")
        else:
            s += _lit(f, f"{base}/{cle}", nom)
    for cle, nom in zones:
        s += _cadres(f, f"{base}/{cle}", nom)
    if part_v:
        s += C.ligne("part de V", _num(f, f"{base}/part_V", "%"), "", AIDE_PART_V)
    return s


def _rubrique_armatures(f: Dict) -> str:
    t = f["type"]
    if t == S8:
        s = (_titre("Poutre support châssis")
             + _section_armatures(f, "support", True, (("etriers_extremites", "étr. extrémités"), ("etriers_centre", "étr. centre")), True)
             + _titre("Semelle sur pieux") + _section_armatures(f, "semelle", True, (("etriers", "étriers"),), False))
    elif t == C4:
        s = _titre("Poutre") + _section_armatures(f, "poutre", False, (("etriers_appui", "étr. appui"), ("etriers_travee", "étr. travée")), True)
    else:
        s = (_titre("Poutre secondaire") + _section_armatures(f, "secondaire", False, (("etriers", "étriers"),), True)
             + _titre("Poutre primaire") + _section_armatures(f, "primaire", False, (("etriers", "étriers"),), False))
    s += _titre("Renfort au droit du pieu")
    s += C.ligne2(("brins", _num(f, "renfort/brins", "", True), "Brins verticaux des étriers de renfort posés au droit de chaque pieu."),
                  ("Ø", _diametre(f, "renfort/d"), ""))
    if t == S8:
        s += C.ligne("brins de base", _num(f, "renfort/brins_base", "", True), "", "Brins des étriers courants de la semelle qui passent au droit du pieu "
                     "et que le classeur compte avec le renfort.")
    s += _titre("Perfos (effort horizontal)")
    s += _lit(f, "perfos", "par pied", "Barres de couture qui reprennent l'effort horizontal sous chaque pied de la grue.")
    return s


def _resume_charges(f: Dict) -> str:
    ch = f["charges"]
    if f["type"] == S8:
        p = max(c[k] for c in ch["cas"] for k in EN.PIEDS)
        return f"{len(ch['cas'])} cas · P max {_g(p)} kN"
    return f"P {_g(ch['P'])} kN · H {_g(max(ch['H_s'], ch['H_hs']))} kN"


def _resume_geometrie(f: Dict) -> str:
    t = f["type"]
    if t == S8:
        return f"châssis {_g(f['chassis'])} · a {_g(f['dist_a'])} · b {_g(f['dist_b'])} cm"
    if t == C4:
        return f"châssis {_g(f['chassis'])} · poutre {_g(f['poutre']['b'])} × {_g(f['poutre']['h'])} cm"
    return f"châssis {_g(f['chassis'])} · entraxe {_g(f['pieux']['entraxe'])} cm"


def _resume_tolerances(f: Dict) -> str:
    t = f["type"]
    es = (f["e_support"], f["e_semelle"]) if t == S8 else (f["e"],) if t == C4 else (f["e_secondaire"], f["e_primaire"])
    return "e = " + " et ".join(_g(e) for e in es) + " cm"


def formulaire(etat: Dict, r: Optional[Dict] = None) -> str:
    """Les six rubriques de la fondation ouverte. ``r`` : son calcul (``MO.verifier``), refait s'il manque."""
    f = _fondation(etat)
    r = r or MO.verifier(f)
    n_arm = sum(1 for a in r["armatures"] if a.get("saisie"))
    return (C.section(1, "Fondation", C.esc(EN.COURTS_TYPE[f["type"]]), _rubrique_fondation(f), True, "fo")
            + C.section(2, "Matériaux", f"{C.esc(f['beton'])} · {C.esc(f['acier'])}", _rubrique_materiaux(f, r), False, "ma")
            + C.section(3, "Géométrie", _resume_geometrie(f), _rubrique_geometrie(f), True, "ge")
            + C.section(4, "Tolérances d'exécution", _resume_tolerances(f), _rubrique_tolerances(f), False, "to")
            + C.section(5, "Charges de la grue", _resume_charges(f), _rubrique_charges(f), False, "ch")
            + C.section(6, "Armatures choisies", f"{n_arm} postes", _rubrique_armatures(f), False, "ar"))


def _dims(f: Dict) -> str:
    """« · 8 pieux · semelles » : le schéma d'une fondation dans la liste de la note (HTML)."""
    return f"· {C.esc(EN.COURTS_TYPE.get(f.get('type'), ''))}"


def arbre(etat: Dict, r: Optional[Dict] = None) -> str:
    """La colonne de gauche COMPACTE : la ligne de la fondation ouverte (son nom, la liste, copier, supprimer), puis
    ses rubriques."""
    elements = [(f.get("nom") or f"Fondation {i + 1}", _dims(f), False) for i, f in enumerate(etat["fondations"])]
    return E.arbre(GENRE, elements, etat["actif"], formulaire(etat, r), "fondation", compact=True)


# ------------------------------------------------------------------ le résultat
def _court(v: Dict) -> str:
    """Le nom court d'une vérification : la fin de sa clé (« support.inf » → « Armatures inférieures »)."""
    return COURTS.get(v["cle"].rsplit(".", 1)[-1], v["nom"])


_CADRES = re.compile(r"^(\d+) brins Ø(\d+) – (\d+(?:[.,]\d+)?) mm$")
_RENFORT = re.compile(r"^(\d+) brins Ø(\d+) \+ (\d+) brins Ø\d+ de base$")


def _piece(v: Dict) -> str:
    """La pièce d'une vérification, en court pour la 1ʳᵉ ligne d'un résultat (HTML) : « 6 brins Ø12 – 100 mm » →
    « 6 br. Ø12/100 » ; l'écriture entière reste dans la fenêtre de calcul."""
    p = v["piece"]
    m = _CADRES.match(p)
    if m:
        return f"{m.group(1)}{C.NBSP}br. Ø{m.group(2)}/{m.group(3)}"
    m = _RENFORT.match(p)
    if m:
        return f"{m.group(1)} + {m.group(3)}{C.NBSP}br. Ø{m.group(2)}"
    return C.esc(p.replace("brins horizontaux des étriers", "brins horizontaux"))


def _libelles(v: Dict) -> Tuple[str, str]:
    """Ce qu'une vérification compare : (la grandeur calculée, la grandeur disponible)."""
    fin = v["cle"].rsplit(".", 1)[-1]
    if fin.startswith("etriers") and v["unite"] == "mm":
        return COMPARE_ETRIERS
    return COMPARE.get(fin, COMPARE_DEFAUT)


def _comparaison(v: Dict, etat: bool = True) -> str:
    """« néc. 2 458 ≤ posé 2 513 mm² » ; ``etat`` : suivi de l'état que le classeur donne quand ce n'est pas « OK »."""
    if v["valeur"] is None or v["limite"] is None:
        return C.esc(v.get("texte") or "—")
    a, b = _libelles(v)
    signe = "≤" if v["valeur"] <= v["limite"] else "&gt;"
    s = f"{a} {_n(v['valeur'], v['unite'])} {signe} {b} {_nu(v['limite'], v['unite'])}"
    if etat and v["etat_classeur"] != K.OK:
        s += f' · <b class="fg-mot {_niv(v)}">{C.esc(v["etat_classeur"])}</b>'
    return s


def _nom_bloc(b: Dict) -> str:
    return COURTS_BLOC.get(b["cle"], b["nom"])


def _a_regarder(r: Dict) -> List[str]:
    """Ce qui n'est pas « OK » dans le classeur (texte) : les vérifications LIMITE, A VERIFIER ou PAS OK, et le
    cisaillement en zone 3 ou au-delà (« cisaillement en zone 3 : poutre support, semelle sur pieux »)."""
    out = [f"{_court(v).lower()} ({_nom_bloc(next(b for b in r['blocs'] if b['cle'] == v['bloc'])).lower()}) : {v['etat_classeur']}"
           for v in r["verifications"] if v["etat"] != MO.OK]
    zones = {}
    for b in r["blocs"]:
        for i in b["informations"]:
            if i["etat"] in (MO.WARN, MO.NOK) and i.get("zone"):
                zones.setdefault(i["zone"].lower(), []).append(_nom_bloc(b).lower())
    return out + [f"cisaillement en {z} : {', '.join(noms)}" for z, noms in zones.items()]


def _verdict(r: Dict) -> str:
    g = r["gouvernant"]
    if g is None:
        return ""
    v = next(x for x in r["verifications"] if x["cle"] == g["cle"])
    bloc = next(b for b in r["blocs"] if b["cle"] == v["bloc"])
    suite = _a_regarder(r)
    regard = f'<br><span class="fg-reg">À regarder : {C.esc(" ; ".join(suite))}</span>' if suite else ""
    return (f'<div class="verdict" data-calcul="{C.esc(v["cle"])}" title="Voir le calcul">{C.chip(C.pct(r["taux_max"]), _niv_res(r), xl=True)}'
            f'<div class="gov"><b>{C.esc(_nom_bloc(bloc))} · {C.esc(_court(v).lower())}</b>{_comparaison(v, etat=False)}<br>'
            f'état du classeur : {C.esc(v["etat_classeur"])}{regard}</div></div>')


def _efforts(r: Dict, b: Dict) -> str:
    """La ligne d'efforts d'un bloc : la pièce, et les sollicitations que le classeur retient pour elle."""
    s = C.esc(b["piece"])
    el = next((e for e in r["elements"] if e["cle"] == b["cle"]), None)
    if el is None:
        return s
    d = el["dimensionnant"]
    for sym, cle, unite in (("M<sub>max</sub>", "M_max", "kN·m"), ("M<sub>min</sub>", "M_min", "kN·m"), ("V<sub>max</sub>", "V_max", "kN")):
        if d.get(cle) is not None:
            s += f" · {sym}{C.NBSP}{_nu(d[cle], unite)}"
    return s


def _info_txt(i: Dict) -> str:
    """Une information du classeur (une valeur, une prescription), en HTML."""
    val = f"{C.symb_math(i['nom'])} = {_nu(i['valeur'], i.get('unite', ''))}" if i.get("valeur") is not None else C.symb_math(i["nom"])
    return val + (f" · {C.esc(i['texte'])}" if i.get("texte") else "")


def _cle_info(b: Dict, k: int) -> str:
    return f"{b['cle']}.i{k}"


def _blocs(r: Dict) -> str:
    """Un bloc par partie du classeur ; puis les contrôles de géométrie et les écarts relevés dans le classeur."""
    gov = r["gouvernant"]["cle"] if r["gouvernant"] else None
    out, i = "", 0
    for i, b in enumerate(r["blocs"]):
        rows, pied = "", ""
        for v in b["verifications"]:
            taux = C.pct(v["taux"]) if v["taux"] is not None else None
            rows += C.ligne_verif(_court(v), _comparaison(v), taux, _niv(v), calcul=v["cle"], piece=_piece(v), gov=v["cle"] == gov,
                                  etat_txt=v["etat_classeur"])
        for k, info in enumerate(b["informations"]):
            if "formule" in info:
                rows += C.ligne_verif(info["nom"], _info_txt(info).split(" = ", 1)[-1], None, NIVEAUX_ETAT[info["etat"]],
                                      calcul=_cle_info(b, k), etat_txt=(info.get("zone") or "valeur").capitalize())
            else:
                pied += C.aide(_info_txt(info))
        if b["cle"] == "support" and "tableau_cas" in r:      # les grands tableaux du classeur, résumés
            pied += (f'<button type="button" class="fg-lien" data-calcul="cas" title="Pour chaque position de tolérance : le cas de la grue et '
                     f'l\'orientation qui gouvernent">{C.icon("file-text")}Cas de la grue : ce qui gouverne</button>')
        taux = [v["taux"] for v in b["verifications"] if v["taux"] is not None]
        etat_txt = MOTS_BLOC[b["etat"]] + (f" · {_pct(max(taux))}" if taux else "")
        out += E.bloc(i, _nom_bloc(b), CLASSES_BLOC[b["etat"]], etat_txt, _efforts(r, b), rows, f'<div class="fg-infos">{pied}</div>' if pied else "",
                      theme=True)
    n = len(r["blocs"])
    if r["controles"]:
        rows = ""
        for c in r["controles"]:
            tete, _, _ = c["texte"].partition(" ≥ ") if " ≥ " in c["texte"] else c["texte"].partition(" ≤ ")
            signe = "≥" if " ≥ " in c["texte"] else "≤"
            cmp_ = f"{_g(c['valeur'])} {signe if c['ok'] else ('&lt;' if signe == '≥' else '&gt;')} {_g(c['limite'])}{C.NBSP}{C.esc(c['unite'])} · {C.esc(c['texte'])}"
            rows += C.ligne_verif(tete[:1].upper() + tete[1:], cmp_, None, "p" if c["ok"] else "r", etat_txt="OK" if c["ok"] else "Non respecté")
        ko = sum(1 for c in r["controles"] if not c["ok"])
        out += E.bloc(n, "Contrôles de géométrie", "nok" if ko else "ok", f"{ko} non respecté{'s' if ko > 1 else ''}" if ko else "Respectés",
                      "les conditions que le classeur vérifie sur les dimensions saisies", rows, theme=True)
        n += 1
    if r["ecarts"]:
        rows = ""
        for e in r["ecarts"]:
            niv = "r" if e["change_verdict"] else NIVEAUX_GRAVITE.get(e["gravite"], "c")
            u = f"{C.NBSP}{C.esc(e.get('unite') or '')}"
            if e.get("classeur") is not None and e.get("redemontre") is not None:
                cmp_ = f"classeur {_val_ecart(e['classeur'])}{u} · redémontré {_val_ecart(e['redemontre'])}{u}"
            elif e.get("classeur") is not None:
                cmp_ = f"classeur {_val_ecart(e['classeur'])}{u}"
            else:
                cmp_ = C.esc(EC.LIBELLES_GRAVITE.get(e["gravite"], ""))
            bloc = next((_nom_bloc(b) for b in r["blocs"] if b["cle"] == e["bloc"]), "")
            rows += C.ligne_verif(f"{e['code']} · {bloc}" if bloc else e["code"], f"{C.esc(e['titre'])} — {cmp_}", None, niv, calcul="ecarts",
                                  etat_txt="change le verdict" if e["change_verdict"] else EC.LIBELLES_GRAVITE.get(e["gravite"], "écart"))
        out += E.bloc(n, "Écarts relevés dans le classeur", "inc", f"{len(r['ecarts'])} à connaître",
                      "le module reproduit le classeur tel qu'il est : ces points sont signalés, jamais corrigés en silence", rows, theme=True)
    return out


def _export(r: Dict) -> str:
    """Le pied du résultat : les fichiers à télécharger. L'export vers RBAR n'est proposé que s'il peut partir."""
    ctl = X.controle(r)
    boutons = ""
    if ctl["pret"]:
        boutons += C.bouton_telecharger("rbar", "RBAR .json")
    boutons += C.bouton_telecharger("dxf", "Plan .dxf") + C.bouton_telecharger("tableau", "Armatures .csv")
    if r["fondation"]["type"] == S8:
        boutons += C.bouton_telecharger("cas", "Cas .csv")
    bloque = ""
    if not ctl["pret"]:
        bloque = C.message("<b>Export vers RBAR bloqué</b> : " + C.esc(" ; ".join(ctl["bloquants"])) + ".", grave=True)
    return (f'{bloque}<div class="fg-exp"><button type="button" class="btn sm" data-calcul="export" title="Ce qui part, ce qui manque, ce qui bloque">'
            f'{C.icon("file-text")}Contrôle</button>{boutons}</div>')


# ------------------------------------------------------------------ les fenêtres de calcul
def _fx(fx: Dict) -> str:
    """Une formule du classeur : « symbole = expression = valeurs = résultat », puis la CELLULE dont elle reproduit la
    valeur. Le gabarit et ses termes viennent du moteur ; ``ecriture`` les écrit, comme dans la note PDF (le test relit
    la chaîne imprimée)."""
    parts, res = W.morceaux(fx)
    h = f'<span class="s">{C.symb_math(parts[0])}</span>' + "".join(f'<span class="eq">=</span><span>{C.symb_math(p)}</span>' for p in parts[1:])
    cel = f'<span class="fg-cel" title="Cellule du classeur">{C.esc(fx["cellule"])}</span>' if fx.get("cellule") else ""
    return f'<div class="fx">{h}<span class="eq">=</span><span class="r">{C.esc(res)}</span>{cel}</div>'


def _groupe(titre: str, formules: Sequence[Dict], sous: str = "") -> str:
    if not formules:
        return ""
    st = f'<span class="ck-comb">{C.esc(sous)}</span>' if sous else ""
    return f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}{st}</div>{"".join(_fx(f) for f in formules)}</div>'


def _hyps(r: Dict) -> List[str]:
    """Les hypothèses d'une fenêtre, en pastilles : la méthode, les matériaux et les contraintes de la base de données."""
    m = r["materiaux"]
    z = " / ".join(C.fr(x, 2) for x in m["tau_zones"])
    return ["Méthode du classeur : <b>contraintes admissibles</b>",
            f"Béton <b>{C.esc(m['beton'])}</b> · R'<sub>wk</sub> = {_g(m['Rwk'])} · σ'<sub>b</sub> = {C.fr(m['sigma_b'], 2)} N/mm²",
            f"Acier <b>{C.esc(m['acier'])}</b> · σ<sub>adm</sub> = {C.fr(m['sigma_adm'], 2)} N/mm² · μ = {C.fr(m['mu'], 4)}",
            f"Cisaillement : zones 1 / 2 / 3 jusqu'à {z} N/mm²",
            f"Classeur de référence : <b>{C.esc(r['classeur'])}</b>"]


def _ligne_tableau(v: Dict, gov: bool) -> Dict:
    a, b = _libelles(v)
    formule = C.symb_math(W.litteral(v["formules"][0])) if v["formules"] and v["formules"][0]["gabarit"] else ""
    return {"nom": C.esc(_court(v)), "sous": f"{C.esc(v['piece'])} · classeur : {C.esc(v['etat_classeur'])} ({C.esc(v['cellule_etat'])})",
            "formule": formule, "valeur": f"{a} = {_nu(v['valeur'], v['unite'])}", "limite": f"{b} = {_nu(v['limite'], v['unite'])}",
            "taux": C.pct(v["taux"]) if v["taux"] is not None else None, "niveau": _niv(v), "gov": gov}


def _pied_fenetre(r: Dict) -> str:
    return (f'<div class="ck-autres fg-pied">Formules, unités et arrondis : ceux des {C.esc(r["reference"])}. Chaque ligne porte la cellule de la '
            f'fiche dont elle reproduit la valeur ; les écarts relevés sont dans le bloc « Écarts ».</div>')


def _fenetre_ecarts(r: Dict, nom: str, hyps: List[str]) -> str:
    lignes, detail = [], ""
    for e in r["ecarts"]:
        niv = "r" if e["change_verdict"] else NIVEAUX_GRAVITE.get(e["gravite"], "c")
        bloc = next((b["nom"] for b in r["blocs"] if b["cle"] == e["bloc"]), "")
        u = f"{C.NBSP}{C.esc(e.get('unite') or '')}"
        lignes.append({"nom": f"{C.esc(e['code'])} · {C.esc(e['titre'])}", "sous": C.esc(bloc), "formule": C.esc(e.get("cellule") or ""),
                       "valeur": f"{_val_ecart(e['classeur'])}{u}" if e.get("classeur") is not None else "—",
                       "limite": f"{_val_ecart(e['redemontre'])}{u}" if e.get("redemontre") is not None else "—",
                       "niveau": niv, "etat_txt": "change le verdict" if e["change_verdict"] else EC.LIBELLES_GRAVITE.get(e["gravite"], "écart")})
        detail += f'<div class="ck-g"><div class="ck-t">{C.esc(e["code"])} · {C.esc(e["titre"])}</div><div class="fg-msg">{C.symb_math(e["message"])}</div></div>'
    pied = ('<div class="ck-autres fg-pied">Le module calcule comme le classeur : aucun de ces écarts n\'est corrigé. « Classeur » est la valeur de la '
            'fiche, « redémontré » celle que donne la même vérification refaite à partir de la statique (poutre sur deux appuis).</div>')
    return C.fenetre_calcul("ecarts", f"{nom} · écarts relevés dans le classeur", f"classeur {C.esc(r['classeur'])} · {len(r['ecarts'])} point(s)", hyps,
                            lignes, detail, pied, colonnes=("Écart", "Cellule", "Classeur", "Redémontré", "Effet"))


def _fenetre_export(r: Dict, nom: str, hyps: List[str]) -> str:
    ctl = X.controle(r)
    mots = {MO.OK: ("p", "OK"), MO.WARN: ("v", "LIMITE"), MO.NOK: ("r", "PAS OK"), MO.INFO: ("c", "prescription")}
    lignes = [{"nom": C.esc(li["element"]), "sous": "", "formule": C.esc(li["poste"]), "valeur": C.esc(li["valeur"]), "limite": C.esc(li["cellules"] or "—"),
               "niveau": mots[li["etat"]][0], "etat_txt": mots[li["etat"]][1]} for li in ctl["lignes"]]

    def liste(titre, items):
        return f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}</div><ul class="fg-li">{"".join(f"<li>{C.esc(x)}</li>" for x in items)}</ul></div>' if items else ""
    detail = (liste("Ce qui bloque l'export", ctl["bloquants"]) + liste("À regarder avant d'envoyer", ctl["a_regarder"])
              + liste("Ce que le calcul ne fixe pas (à compléter par le dessinateur)", ctl["manques"])
              + liste("Ce que RBAR ne reprend pas d'un fichier", X.LIMITES_RBAR) + liste("Écarts relevés dans le classeur", ctl["ecarts"]))
    etat = "prêt à partir" if ctl["pret"] else "bloqué"
    pied = ('<div class="ck-autres fg-pied">Les valeurs exportées sont celles du calcul : l\'export ne recalcule rien. Le fichier RBAR porte les modèles '
            'de poutre (palette › modèles › « Importer JSON… »), les types de pieux et de massifs, l\'implantation en mm et le tableau des armatures.</div>')
    return C.fenetre_calcul("export", f"{nom} · contrôle avant export", f"vers RBAR (AutoCAD) · {etat}", hyps, lignes, detail, pied,
                            colonnes=("Élément", "Poste", "Valeur exportée", "Cellules de saisie", "État"))


GRANDEURS_CAS = (("RA", "R_A,max", "kN"), ("RB", "R_B,max", "kN"), ("RB_resp", "R_B respectif", "kN"), ("RB_symm", "R_B symétrique", "kN"),
                 ("V_max", "V_max", "kN"), ("V_central", "V_central", "kN"), ("M_max", "M_max", "kN·m"), ("M_max_eq", "M_max,éq", "kN·m"))


def _colonne(x: Dict) -> str:
    """La colonne du classeur qui donne un maximum (« CAS 1.a-B/A ») ; « — » quand aucune ne le donne (une grandeur
    que le classeur ne calcule que pour les cas symétriques, sans cas symétrique)."""
    return (x.get("libelle") or "—").strip("()")


def _fenetre_cas(r: Dict, nom: str, hyps: List[str]) -> str:
    """Les quatre grands tableaux du classeur (8 pieux sur semelles), résumés : pour chaque position de tolérance, la
    portée et, pour chaque grandeur, son maximum sur les 48 colonnes (6 cas × 8 orientations) et la colonne qui le
    donne. Toutes les colonnes sont dans le fichier « Cas de la grue (.csv) »."""
    lignes, detail = [], ""
    pos = {p["cle"]: p for el in r["elements"] if el["cle"] == "support" for p in el["positions"]}
    for t in r["tableau_cas"]:
        g, mx = t["geometrie"], t["max"]
        p = pos.get(t["sous_cas"], {})
        lignes.append({"nom": f"CAS {C.esc(t['sous_cas'])}", "sous": C.esc(p.get("texte", "")), "formule": f"L = {C.fr(g['L'], 2)}{C.NBSP}m",
                       "valeur": f"M = {_nu(mx['M_max']['valeur'], 'kN·m')}", "limite": f"V = {_nu(mx['V_max']['valeur'], 'kN')}",
                       "niveau": "c", "etat_txt": _colonne(mx["M_max"])})
        corps = "".join(f'<div class="fx"><span class="s">{C.symb_math(sym)}</span><span class="eq">=</span><span class="r">{_nu(mx[k]["valeur"], u)}</span>'
                        f'<span class="fg-cel">{C.esc(_colonne(mx[k]))}</span></div>' for k, sym, u in GRANDEURS_CAS if k in mx)
        detail += (f'<div class="ck-g"><div class="ck-t">CAS {C.esc(t["sous_cas"])}<span class="ck-comb">feuille « {C.esc(t["feuille"])} » · '
                   f'{len(t["colonnes"])} colonnes</span></div>{corps}</div>')
    pied = ('<div class="ck-autres fg-pied">Chaque position de tolérance est calculée pour chaque cas de la grue dans les huit orientations (A/B, B/A, '
            'B/C, C/B, C/D, D/C, D/A, A/D) : le maximum de chaque grandeur et la colonne qui le donne. Le détail de toutes les colonnes se télécharge '
            '(« Cas de la grue (.csv) ») pour être comparé aux feuilles du classeur.</div>')
    return C.fenetre_calcul("cas", f"{nom} · cas de la grue et positions de tolérance", "poutre support châssis · les grands tableaux du classeur", hyps,
                            lignes, detail, pied, colonnes=("Position", "Portée", "Moment max", "Tranchant max", "Colonne qui gouverne M"))


def _volets(r: Dict, nom: str) -> str:
    """Les fenêtres de calcul : une par vérification (son tableau, les sollicitations de sa pièce, ses formules), une
    par valeur calculée (τ), « tout le calcul » de chaque bloc (s<i>), les écarts, le contrôle avant export et, pour
    les 8 pieux sur semelles, les cas de la grue."""
    hyps, pied = _hyps(r), _pied_fenetre(r)
    gov = r["gouvernant"]["cle"] if r["gouvernant"] else None
    out = ""
    for i, b in enumerate(r["blocs"]):
        base = f"{C.esc(b['nom'])} · {C.esc(b['piece'])}"
        soll = _groupe("Sollicitations", b["sollicitations"], b["nom"])
        tout = soll
        for v in b["verifications"]:
            g = _groupe(v["nom"], v["formules"])
            tout += g
            out += C.fenetre_calcul(v["cle"], f"{nom} · {C.esc(_court(v).lower())}", base, hyps, [_ligne_tableau(v, True)], soll + g, pied)
        for k, info in enumerate(b["informations"]):
            if "formule" not in info:
                continue
            g = _groupe(info["nom"], [info["formule"]])
            tout += g
            niv = NIVEAUX_ETAT[info["etat"]]
            ligne = {"nom": C.symb_math(info["nom"]), "sous": C.esc(info.get("texte") or ""), "formule": C.symb_math(W.litteral(info["formule"])),
                     "valeur": _nu(info["valeur"], info.get("unite", "")), "limite": "—", "niveau": niv,
                     "etat_txt": (info.get("zone") or "valeur").capitalize()}
            out += C.fenetre_calcul(_cle_info(b, k), f"{nom} · {C.symb_math(info['nom'])}", base, hyps, [ligne], soll + g, pied)
        lignes = [_ligne_tableau(v, v["cle"] == gov) for v in b["verifications"]]
        out += C.fenetre_calcul(f"s{i}", f"{nom} · {C.esc(b['nom'].lower())}", f"{base} · tout le calcul", hyps, lignes, tout, pied)
    n = len(r["blocs"])
    if r["controles"]:
        lignes = [{"nom": C.esc(c["texte"]), "sous": "", "formule": C.esc(c["cellule"]), "valeur": f"{_g(c['valeur'])}{C.NBSP}{C.esc(c['unite'])}",
                   "limite": f"{_g(c['limite'])}{C.NBSP}{C.esc(c['unite'])}", "niveau": "p" if c["ok"] else "r",
                   "etat_txt": "OK" if c["ok"] else "Non respecté"} for c in r["controles"]]
        out += C.fenetre_calcul(f"s{n}", f"{nom} · contrôles de géométrie", "les conditions que le classeur vérifie sur les dimensions", hyps, lignes, "",
                                pied, colonnes=("Condition", "Cellule", "Valeur", "Limite", "État"))
        n += 1
    if r["ecarts"]:
        out += _fenetre_ecarts(r, nom, hyps)
        out += _fenetre_ecarts(r, nom, hyps).replace('id="ck-ecarts"', f'id="ck-s{n}"', 1)
    out += _fenetre_export(r, nom, hyps)
    if "tableau_cas" in r:
        out += _fenetre_cas(r, nom, hyps)
    return out


# ------------------------------------------------------------------ les vues
def _entete(r: Dict, titre: str, legende: str, cle: str) -> str:
    """L'en-tête technique d'une vue : le schéma et les matériaux ; les charges ; les tolérances ; à droite, pour une
    poutre ou une semelle, les sollicitations que le classeur retient."""
    f, m = r["fondation"], r["materiaux"]
    el = next((e for e in r["elements"] if "mv_" + e["cle"] == cle), None)
    bande = None
    if el is not None:
        d = el["dimensionnant"]
        bande = [(s, x, d[k], u) for s, x, k, u in (("M", "max", "M_max", "kN·m"), ("M", "min", "M_min", "kN·m"), ("V", "max", "V_max", "kN"))
                 if d.get(k) is not None]
    pieux = next(p for p in r["pieces"] if p["cle"] == "pieux")
    lignes = [f"{_resume_charges(f)} · {_resume_tolerances(f)}",
              f"{pieux['nombre']} pieux Ø{_g(pieux['D'])} cm" + (f" · entraxe {_g(pieux['entraxe'])} cm" if pieux.get("entraxe") else "")
              if el is None else f"{el['nom']} sur deux appuis · {len(el['positions'])} positions de tolérance"]
    sous = f"{C.esc(EN.COURTS_TYPE[f['type']])} · {C.esc(m['beton'])} · {C.esc(m['acier'])}"
    return E.entete(titre, sous, lignes, bande, legende)


def _vues(r: Dict, ident: str) -> List[Dict]:
    """La vue en plan puis, pour chaque poutre et chaque semelle, son schéma et ses diagrammes M et V."""
    return [{"cle": cle, "titre": titre, "court": court, "svg": SC.svg(scene, f"{ident}_{cle}", titre), "entete": _entete(r, titre.upper(), legende, cle),
             "legende": legende, "echelle": echelle} for cle, titre, court, echelle, scene, legende in SC.vues(r)]


def _calculer(f: Dict) -> Dict:
    try:
        return MO.verifier(f)
    except (ValueError, KeyError, TypeError) as e:              # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(e)}


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE (l'accueil, la liste des notes) : sa fondation la plus chargée."""
    vals = [(x["taux_max"], i) for i, x in enumerate(resultats) if x.get("taux_max") is not None]
    if not vals:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    t, i = max(vals)
    x = resultats[i]
    pire = next((k for k in (MO.NOK, MO.WARN) if any(y.get("statut") == k for y in resultats)), MO.OK)
    niv = NIVEAUX_ETAT[pire] if pire != MO.OK else C.niveau(t)
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1],
            "gouvernant": f"{etat['fondations'][i].get('nom') or 'Fondation'} · {x['gouvernant']['nom'].lower()}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """Rien : les fondations de la note sont la LIGNE de l'élément de la colonne de gauche (``arbre``)."""
    return ""


def resultat(etat: Dict) -> Dict:
    """Le résultat de la fondation ouverte : la tête, le verdict, les blocs, l'export ; les fenêtres de calcul, les
    vues, la colonne de gauche, le résumé de la note."""
    f = _fondation(etat)
    r = MO.verifier(f)
    resultats = [r if i == etat["actif"] else _calculer(x) for i, x in enumerate(etat["fondations"])]
    nom = C.esc(f.get("nom") or "Fondation")
    tete = E.tete_resultats(nom, f"{C.esc(EN.COURTS_TYPE[f['type']])} · {C.esc(f['beton'])} · {C.esc(f['acier'])}")
    pied = f'<div class="pfoot">{_export(r)}<div class="legend">{C.legende()}</div></div>'
    resume = resume_note(etat, resultats)
    statut = C.statut(resume["niveau"]) if len(resultats) > 1 else ""
    corps = f'<div class="pbody">{tete}{_verdict(r)}<div class="et-blocs">{_blocs(r)}</div></div>{pied}'
    return {"resultat": corps, "volets": _volets(r, nom), "vues": _vues(r, f"fg{etat['actif']}"), "phrase": "", "statut_html": statut,
            "resume": resume, "donnees": arbre(etat, r), "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ les fichiers à télécharger
def telecharger(etat: Dict, quoi: str, infos: Optional[Dict] = None) -> Optional[Dict]:
    """« rbar » : le fichier d'échange vers RBAR (JSON ; refusé avec un message quand une vérification du classeur est
    PAS OK) ; « dxf » : le plan de principe (DXF R12, en cm) ; « tableau » : le tableau des armatures (CSV) ; « cas » :
    toutes les colonnes des grands tableaux des cas de la grue (CSV, 8 pieux sur semelles)."""
    f = _fondation(etat)
    nom = f.get("nom") or "Fondation de grue"
    if quoi not in ("rbar", "dxf", "tableau", "cas"):
        return None
    r = MO.verifier(f)
    if quoi == "rbar":
        return {"nom": C.nom_fichier(f"{nom} - RBAR", "json"), "type": "application/json", "texte": X.rbar_texte(r, infos)}
    if quoi == "dxf":
        return {"nom": C.nom_fichier(f"{nom} - plan de principe", "dxf"), "type": "text/plain", "texte": X.dxf(r, infos)}
    if quoi == "tableau":
        return {"nom": C.nom_fichier(f"{nom} - armatures", "csv"), "type": "text/csv", "texte": X.tableau(r)}
    if "tableau_cas" not in r:
        raise ValueError("Ce schéma n'a pas de tableau des cas de la grue (une seule charge par pied).")
    return {"nom": C.nom_fichier(f"{nom} - cas de la grue", "csv"), "type": "text/csv", "texte": X.tableau_cas(r)}

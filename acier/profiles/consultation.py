# -*- coding: utf-8 -*-
"""Ce que la page « Catalogue des profilés » fait du catalogue, SANS Streamlit : la liste affichée (familles,
recherche, filtres numériques, séries courantes, profilés personnalisés, tri), les lignes du tableau, le compteur, la
fiche d'un profilé, la comparaison, le profilé personnalisé, les réglages (défauts livrés, toutes les familles…).

Extrait MOT POUR MOT de l'écran Streamlit ``modules/tableau_profiles.py`` (version 1.0), qui reste la RÉFÉRENCE : chaque
fonction reprend l'expression de l'écran, dans le même ordre (le site ``api_web/catalogue.py`` l'appelle dans le
navigateur, où ni Streamlit ni pandas n'existent). La preuve est ``tests/test_catalogue_moteur.py`` : l'écran est
rejoué par le vrai Streamlit (AppTest) sur des dizaines d'états et ce module donne les mêmes listes, les mêmes
nombres, les mêmes textes, les mêmes fichiers exportés.

Ce qui n'est PAS dans l'écran et que le site ajoute est rangé à part et nommé : ``refus_du_site`` (une saisie que
l'écran ramène en silence à une autre valeur, ou qui le fait tomber, est refusée avec un message),
``avertissements_perso`` (un résultat que l'écran affiche sans dire qu'il n'a plus de sens est affiché pareil, et
écrit), ``reserve_torsion`` (le I_t d'un PRS, non conservatif, affiché pareil et écrit). Voir
docs/acier/CATALOGUE_MOTEUR.md.

Bibliothèque standard + le paquet ``acier.profiles`` (numpy pour le calcul d'un profilé personnalisé).
"""
import csv
import io
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import catalogue as CAT
from . import familles as F

VERSION = "1.0"                 # la version affichée par l'écran (modules/tableau_profiles.py, VERSION)
N_FILTRES = 3                   # trois filtres numériques (N_FILTRES de l'écran)
# l'aide du bouton « Séries courantes seulement » de l'écran
AIDE_COURANTS = "IPE, HE A/B/M, UPE, UPN, cornières, tubes ; masque IPE A/O, IPN, HE AA, HL, HD, HP."
# les dimensions proposées à l'ouverture du profilé personnalisé (``defauts`` de _perso)
DIMS_DEFAUT = {"h": 400.0, "b": 200.0, "tw": 8.0, "tf": 12.0, "t": 6.0, "d": 100.0}
DIM_MIN = 0.1                   # min_value des champs de dimensions de l'écran (mm)
# AJOUT DU SITE (l'écran n'a pas de borne haute) : au-delà, le calcul déborde — h = 1e300 mm lève OverflowError, que
# l'écran n'attrape pas (il n'attrape que ValueError : la page tombe) ; 10 m couvre tout profilé reconstitué réel
DIM_MAX = 10000.0
MAX_COMPARES = 5                # « jusqu'à 5 » (max_selections de l'écran)
# la méthode de I_t des profilés personnalisés à parois (PRS) : celle que familles.calculer écrit dans « methode_It »
METHODE_IT_MINCE = "Σ b·t³/3 (parois minces, sans congé)"
MESSAGE_VIDE = "Aucun profilé : élargir les familles, la recherche ou les filtres."
AIDE_COLONNES_ECRAN = ("Les colonnes vides pour les familles affichées sont masquées d'elles-mêmes. "
                       "Survoler l'en-tête d'une fiche donne la définition.")
AIDE_COMPARER = "Choisir au moins deux profilés du tableau."


# ------------------------------------------------------------------ colonnes et en-têtes
def cles_colonnes() -> List[str]:
    """Les 39 propriétés, dans l'ordre du catalogue (``cles_num`` de l'écran)."""
    return [c[0] for c in CAT.COLONNES]


def entete(c: str) -> str:
    """« Libellé (unité) » — _entete de l'écran."""
    u = CAT.UNITE.get(c, "")
    return f"{CAT.LIBELLE[c]} ({u})" if u else CAT.LIBELLE[c]


def colonnes_ordonnees(choix: Sequence[str]) -> List[str]:
    """Les colonnes retenues, dans l'ordre du catalogue : l'écran réécrit sa liste ainsi à chaque exécution
    (``[c for c in cles_num if c in choix]``), quel que soit l'ordre des clics."""
    return [c for c in cles_colonnes() if c in choix]


# ------------------------------------------------------------------ familles
def toutes_familles() -> List[str]:
    """Les familles présentes au catalogue, groupe par groupe (``toutes`` de l'écran)."""
    return [f for _, fams in CAT.groupes_presents() for f in fams]


def familles_ordonnees(choix: Sequence[str]) -> List[str]:
    """Les familles retenues dans l'ordre des pastilles (_lire_pills) : groupe par groupe, celles du catalogue
    seulement — une famille absente du catalogue (PRS…) n'a pas de pastille et sort de la liste."""
    return [f for f in toutes_familles() if f in choix]


# ------------------------------------------------------------------ réglages
def filtres_vides() -> List[Dict]:
    return [{"cle": "", "min": None, "max": None} for _ in range(N_FILTRES)]


def reglages_initiaux() -> Dict:
    """Les réglages d'ouverture : les préférences enregistrées (acier/donnees/preferences_profiles.json) complétées
    des défauts — _init de l'écran, puis la première exécution (pastilles et colonnes remises dans leur ordre)."""
    prefs = CAT.preferences()
    return {"familles": familles_ordonnees(prefs["familles"]), "colonnes": colonnes_ordonnees(prefs["colonnes"]),
            "tri": prefs["tri"] if prefs["tri"] in cles_colonnes() else "", "decroissant": bool(prefs["decroissant"]),
            "courants": bool(prefs["courants_seulement"]), "filtres": filtres_vides()}


def reglages_livres() -> Dict:
    """« ↩ Rétablir » : les défauts LIVRÉS (PREFS_DEFAUT), décroissant et séries courantes à faux, filtres effacés.
    L'écran les écrit aussi dans le fichier des préférences (pour tout le monde) ; ici rien n'est écrit."""
    d = CAT.PREFS_DEFAUT
    return {"familles": familles_ordonnees([f for f in d["familles"] if f in F.FAMILLES]),
            "colonnes": colonnes_ordonnees([c for c in d["colonnes"] if c in CAT.LIBELLE] or CAT.DEFAUT_COLONNES),
            "tri": d["tri"], "decroissant": False, "courants": False, "filtres": filtres_vides()}


# ------------------------------------------------------------------ la liste
def bornes(filtres: Sequence[Dict]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """{clé: (min, max)} des filtres dont la propriété est choisie — l'expression de l'écran. DÉFAUT DE L'ÉCRAN,
    reproduit : deux filtres sur la MÊME propriété, seul le dernier compte (le premier est perdu sans le dire) ; le
    site refuse donc de choisir deux fois la même propriété (``refus_du_site``)."""
    return {f["cle"]: (f["min"], f["max"]) for f in filtres if f["cle"]}


def liste(familles: Sequence[str], recherche: str, filtres: Sequence[Dict], courants: bool, persos: Sequence[Dict],
          tri: str, decroissant: bool) -> List[Dict]:
    """La liste affichée : le catalogue filtré (familles, recherche, bornes, séries courantes), PUIS les profilés
    personnalisés filtrés par la recherche et les bornes seulement, le tout trié si un tri est choisi."""
    b = bornes(filtres)
    base = CAT.filtrer(CAT.profils(), familles, recherche, b, courants)
    perso = CAT.filtrer(persos, None, recherche, b, False)
    out = base + perso
    return CAT.trier(out, tri, decroissant) if tri else out


def n_persos_affiches(recherche: str, filtres: Sequence[Dict], persos: Sequence[Dict]) -> int:
    """Le nombre de profilés personnalisés retenus (le « · n personnalisés » du compteur)."""
    return len(CAT.filtrer(persos, None, recherche, bornes(filtres), False))


def compteur(n: int, familles: Sequence[str], n_persos: int) -> str:
    """Le compteur au-dessus du tableau, au texte près (markdown de l'écran)."""
    return (f"**{n} profilé{'s' if n > 1 else ''}**"
            + (f" · {len(familles)} famille{'s' if len(familles) > 1 else ''}" if familles else "")
            + (f" · {n_persos} personnalisé{'s' if n_persos > 1 else ''}" if n_persos else ""))


def colonnes_visibles(lst: Sequence[Dict], colonnes: Sequence[str]) -> List[str]:
    """Les colonnes retenues qui portent au moins une valeur dans la liste (cols_vis de l'écran)."""
    return CAT.colonnes_utiles(lst, colonnes)


def lignes_tableau(lst: Sequence[Dict], colonnes: Sequence[str]) -> List[Dict]:
    """Les lignes du tableau (le DataFrame de l'écran, sans pandas) : « Profilé », « Famille », puis chaque colonne
    sous son en-tête « Libellé (unité) », la valeur ARRONDIE aux décimales de la colonne (None si absente)."""
    out = []
    for p in lst:
        ligne = {"Profilé": p["nom"], "Famille": p["famille"]}
        for c in colonnes:
            v = p.get(c)
            ligne[entete(c)] = None if v is None else round(float(v), CAT.DECIMALES.get(c, 2))
        out.append(ligne)
    return out


def texte_tableau(lignes: Sequence[Dict], separateur: str = ";") -> str:
    """Le tableau affiché en texte, cellule pour cellule celui de l'export CSV de l'écran (_csv, sans pandas ni BOM) :
    en-têtes, puis chaque valeur écrite ``str(v)`` avec la virgule décimale, une cellule vide pour une valeur absente,
    fins de ligne CRLF (le module csv). DÉFAUT DE L'ÉCRAN, reproduit : le point est remplacé dans TOUTES les cellules,
    les noms compris — « HD 260x54.1 » s'exporte « HD 260x54,1 », qui n'est plus le nom du catalogue (456 profilés)."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=separateur)
    w.writerow(list(lignes[0].keys()) if lignes else [])
    for ligne in lignes:
        w.writerow([("" if v is None else str(v).replace(".", ",")) for v in ligne.values()])
    return buf.getvalue()


def fiche_json(p: Dict) -> str:
    """La fiche d'un profilé telle que l'écran la télécharge (« ⬇️ Fiche (JSON) ») : l'enregistrement complet."""
    return json.dumps(p, ensure_ascii=False, indent=1)


def nom_fichier_fiche(p: Dict) -> str:
    return f"{p['nom'].replace(' ', '_')}.json"


def fiche_retenue(fiche: Optional[str], noms: Sequence[str]) -> Optional[str]:
    """Le profilé de la fiche : celui retenu s'il est dans la liste, sinon le premier de la liste (et l'écran le
    GARDE : la fiche ne revient pas quand le filtre qui l'avait chassée est retiré)."""
    return fiche if fiche in noms else (noms[0] if noms else None)


def compares_retenus(compare: Sequence[str], noms: Sequence[str]) -> List[str]:
    """Les profilés à comparer : la liste à choix multiples de l'écran ne garde que ceux du tableau ; un profilé qui en
    sort quitte la comparaison (et n'y revient pas)."""
    return [n for n in compare if n in noms]


# ------------------------------------------------------------------ la fiche
def chips_fiche(p: Dict) -> List[str]:
    """Les pastilles sous le nom : le libellé de la famille, sa norme, « personnalisé »."""
    f = F.FAMILLES[p["famille"]]
    return [f["libelle"], f["norme"]] + (["personnalisé"] if p.get("perso") else [])


def fiche(p: Dict) -> List[Tuple[str, List[Tuple[str, str, str, str, str]]]]:
    """Toutes les propriétés d'un profilé, groupe par groupe (_table_fiche) : [(groupe, [(clé, libellé, valeur écrite,
    unité, définition)])] ; un groupe sans valeur est omis, une propriété absente aussi."""
    out = []
    for g in CAT.GROUPES_COLONNES:
        cles = [c for c in CAT.colonnes_du_groupe(g) if p.get(c) is not None]
        if not cles:
            continue
        out.append((g, [(c, CAT.LIBELLE[c], CAT.formater(p[c], c), CAT.UNITE.get(c, ""), CAT.AIDE_COLONNES.get(c, ""))
                        for c in cles]))
    return out


def methode_it(p: Dict) -> Optional[str]:
    """La méthode de I_t (la légende « It : … » de l'écran) ; None si le profilé n'en porte pas."""
    return p.get("methode_It") or None


def comparaison(profils: Sequence[Dict], colonnes: Sequence[str]) -> List[Tuple[str, str, List[str]]]:
    """Le tableau de comparaison (_table_comparaison) : [(clé, « Libellé (unité) », [valeur écrite de chaque profilé])],
    une ligne par colonne dont un profilé au moins porte la valeur. ``colonnes`` : les colonnes visibles, ou les
    colonnes par défaut si aucune ne l'est (l'écran passe ``cols_vis or CAT.DEFAUT_COLONNES``)."""
    out = []
    for c in colonnes:
        if all(p.get(c) is None for p in profils):
            continue
        out.append((c, entete(c), [CAT.formater(p.get(c), c) for p in profils]))
    return out


def colonnes_comparaison(cols_vis: Sequence[str]) -> List[str]:
    return list(cols_vis) or list(CAT.DEFAUT_COLONNES)


# ------------------------------------------------------------------ le profilé personnalisé
def formes_perso() -> List[str]:
    """Les formes d'un profilé personnalisé (PRS, caisson, tube, plat, rond) — les familles « non standard »."""
    return [f for f, d in F.FAMILLES.items() if d.get("non_standard")]


def dims_defaut(fam: str) -> Dict[str, float]:
    return {k: DIMS_DEFAUT[k] for k in F.FAMILLES[fam]["dims"]}


def nom_perso(fam: str, dims: Dict[str, float]) -> str:
    """« PRS 500x200x8x12 » (_nom_perso)."""
    f = F.FAMILLES[fam]
    return f"{fam} " + "x".join(f"{float(dims[k]):g}" for k in f["dims"])


def controle_ecran(fam: str, dims: Dict[str, float]) -> None:
    """Les deux contrôles de l'écran avant le calcul (ValueError, message de l'écran)."""
    if fam in ("PRS",) and 2 * dims["tf"] >= dims["h"]:
        raise ValueError("h doit dépasser 2·tf")
    if fam in ("CAISSON", "TUBE") and 2 * dims["t"] >= min(v for k, v in dims.items() if k in ("h", "b", "d")):
        raise ValueError("t doit être inférieure à la demi-dimension")


def refus_dimension(k: str, x: float) -> Optional[str]:
    """Le message de refus d'UNE dimension (mm), None si elle est admise : un nombre fini, de 0,1 mm (le minimum du
    champ de l'écran, qui remonte en silence une valeur plus petite) à 10 m (au-delà, le calcul de l'écran déborde)."""
    if not (isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)):
        return f"{k} : nombre fini attendu"
    if x < DIM_MIN:
        return f"{k} doit valoir au moins {DIM_MIN:g} mm".replace(".", ",")
    if x > DIM_MAX:
        return f"{k} doit valoir au plus {DIM_MAX:,.0f} mm".replace(",", " ")
    return None


def refus_du_site(fam: str, dims: Dict[str, float]) -> None:
    """Ce que le site REFUSE (ValueError) et que l'écran accepte : une dimension sous 0,1 mm (le champ de l'écran,
    ``min_value=0.1``, la remonte au minimum sans le dire), non finie ou au-delà de 10 m (l'écran calcule jusqu'à
    déborder : h = 1e300 mm le fait tomber). Un fichier .stab abîmé pourrait en porter une."""
    for k in F.FAMILLES[fam]["dims"]:
        msg = refus_dimension(k, dims[k])
        if msg:
            raise ValueError(msg)


def reserve_torsion(p: Dict) -> Optional[str]:
    """DÉFAUT NON CONSERVATIF de l'écran, reproduit et ÉCRIT : un profilé personnalisé à parois (PRS) reçoit
    I_t = Σ b·t³/3 (parois minces), qui SURESTIME la raideur de torsion des plats épais — mesuré contre Saint-Venant
    (différences finies, même contour, grille convergée) : +2 % pour un PRS 400x200x8x12, +10 % pour un
    200x150x15x30. Un I_t trop grand donne un moment critique de déversement trop grand. Le catalogue laminé, lui,
    est calculé par Saint-Venant. Rend la phrase à afficher, None si le profilé n'est pas concerné."""
    if p.get("methode_It") != METHODE_IT_MINCE:
        return None
    return ("I_t par Σ b·t³/3 (parois minces) surestime la raideur de torsion des plats épais : +2 % pour un "
            "PRS 400x200x8x12, +10 % pour un 200x150x15x30 (mesuré contre Saint-Venant) — non conservatif pour le "
            "déversement ; l'écran Streamlit l'affiche sans le dire")


def reserve_cisaillement(p: Dict) -> Optional[str]:
    """RÉSERVE de l'écran, reproduite et ÉCRITE : l'aire de cisaillement A_v,z d'un PRS est celle d'un LAMINÉ
    (A − 2·b·t_f + (t_w + 2·r)·t_f, sans congé : t_w·(h − t_f)), alors que l'EN 1993-1-1 §6.2.6(3)(d) donne
    η·h_w·t_w pour un profilé SOUDÉ. Elle la dépasse de t_f/h_w avec η = 1,0 (+18 % pour un PRS 300x300x20x40 :
    52,0 cm² contre 44,0) ; avec η = 1,2, elle reste en dessous tant que t_f ≤ h/7. Non conservatif si le bureau
    retient η = 1,0. None si le profilé n'est pas un PRS."""
    if p.get("famille") != "PRS" or p.get("Avz") is None:
        return None
    return ("A_v,z suit la formule des laminés, t_w·(h − t_f) ; pour un profilé soudé, l'EN 1993-1-1 §6.2.6(3)(d) "
            "donne η·h_w·t_w, plus faible avec η = 1,0 (+18 % pour un PRS 300x300x20x40) — à reprendre selon le η "
            "retenu ; l'écran Streamlit l'affiche sans le dire")


def avertissements_perso(fam: str, dims: Dict[str, float]) -> List[str]:
    """Ce que l'écran calcule SANS rien dire alors qu'une grandeur n'a plus de sens ; le site calcule la même chose
    (parité) et l'ÉCRIT. Un PRS dont l'âme est au moins aussi épaisse que les semelles sont larges (t_w ≥ b) n'est
    plus un I : l'aire et les inerties restent justes (le contour est l'union des trois plats), mais c_f/t_f devient
    négatif ou nul et A_v,z (formule d'un I) est sans objet — PRS 400x200x300x12 : c_f/t_f = −4,17."""
    out = []
    if fam == "PRS" and float(dims["tw"]) >= float(dims["b"]):
        out.append("âme au moins aussi épaisse que les semelles sont larges (t_w ≥ b) : ce n'est plus un I ; "
                   "c_f/t_f (négatif ou nul) et A_v,z n'ont pas de sens, l'écran Streamlit les affiche sans le dire")
    return out


def calculer_perso(fam: str, dims: Dict[str, float]) -> Dict:
    """Le profilé personnalisé tel que l'écran l'ajoute au tableau : contrôles de l'écran, calcul par le MÊME moteur
    que le catalogue (I_t de paroi mince : ``torsion_exacte=False``), nom, famille, « personnalisé »."""
    controle_ecran(fam, dims)
    p = F.calculer(fam, dims, torsion_exacte=False)
    p.update({"nom": nom_perso(fam, dims), "famille": fam, "courant": False, "dims": dict(dims), "perso": True})
    return p


def ajouter_perso(persos: Sequence[Dict], p: Dict) -> List[Dict]:
    """Un profilé du même nom est remplacé ; le nouveau va en fin de liste."""
    return [q for q in persos if q["nom"] != p["nom"]] + [p]


# ------------------------------------------------------------------ méthode et sources
def legende_page() -> str:
    """La légende sous le titre de la page."""
    return ("Profilés laminés, creux et reconstitués — dimensions normalisées, propriétés calculées depuis le contour exact "
            f"· {len(CAT.profils())} profilés · version {VERSION}")


def methode_et_sources() -> Dict:
    """Le contenu de « Méthode et sources » (méta du fichier, définitions des colonnes, familles)."""
    m = CAT.meta()
    return {"fichier": "acier/donnees/profiles.json", "version": m["version"], "date": m["date"], "nombre": m["nombre"],
            "conventions": m["conventions"], "sources": list(m["sources_dimensions"]), "calcul": m["calcul"],
            "controle": m["controle"], "definitions": [(c, CAT.LIBELLE[c], t) for c, t in CAT.AIDE_COLONNES.items()],
            "familles": [(f, d["libelle"], d["norme"], d["description"]) for f, d in F.FAMILLES.items()]}

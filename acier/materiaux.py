# -*- coding: utf-8 -*-
"""LA table des aciers de construction du site : f_y et f_u selon l'épaisseur nominale.

NBN EN 1993-1-1 ANB:2018, 3.2.1(1) : « L'option “a” est normative » — les valeurs minimales de la limite d'élasticité
(R_eH) et de la résistance à la traction (R_m) sont celles des normes de produit, données par les tableaux 3.3 ANB à
3.12 ANB. Le tableau 3.1 de l'EN (option b) n'est pas admis en Belgique.

Ce fichier est la SEULE source de f_y et de f_u des aciers de construction du site : barres, flambement, cornière,
assemblages, soudures, garde-corps, plancher, Bibliothèque Acier. Un module choisit l'épaisseur qui décide (celle de
la pièce, ou la plus épaisse des parois d'un profilé : la plus petite f_y) et garde ses propres bornes de domaine ; il
ne réécrit jamais une valeur.

Les valeurs ci-dessous sont celles des tableaux, lues sur le texte joint le 16/11/2026 (deux lectures indépendantes
du PDF, concordantes ligne à ligne) ; le texte n'est pas au dépôt (dépôt public, texte sous licence). Seules les lignes
des nuances que le site propose sont reprises. f_u = la valeur MINIMALE de R_m (« 470 à 630 » → 470).

Bornes d'épaisseur (mm) : une bande ``(sup, incluse)`` couvre les épaisseurs t < sup, ou t ≤ sup si ``incluse`` ;
elle commence là où finit la précédente. Les bandes de R_eH sont « ≤ 16 », « > 16 ≤ 40 »… (incluses) ; celles de R_m
diffèrent d'un tableau à l'autre (« < 3 » puis « ≥ 3 ≤ 100 » au tableau 3.3 ANB, « ≤ 3 » puis « > 3 ≤ 100 » au
tableau 3.8 ANB), et sont recopiées telles quelles.

    fy(nuance, t, produit)   f_y (N/mm²) — HorsTable au-delà du tableau
    fu(nuance, t, produit)   f_u (N/mm²)
    ligne(nuance, produit)   {"tableau", "norme", "designation"} : la ligne de l'ANB employée
    source(nuance, t, produit, grandeur)   la référence à imprimer (« tableau 3.3 ANB (EN 10025-2:2004), 16 < t ≤ 40 mm »)
    nominale(nuance)         f_y de la plus petite épaisseur (le nombre du nom : S355 → 355), pour les libellés et
                             les contrôles de domaine — jamais pour une résistance
    paliers(nuance, produit, grandeur)   [(« 16 < t ≤ 40 mm », valeur)] : la ligne entière
    bornes(nuance, produit, grandeur)    [(borne sup., incluse, texte, valeur)] : la même, avec ses bornes chiffrées

Produits : « lamine » (profilés laminés, plats, tôles — EN 10025), « creux_chaud » (profils creux finis à chaud,
EN 10210-1), « creux_froid » (profils creux formés à froid, EN 10219-1).
"""
from typing import Dict, List, Tuple

PRODUITS = ("lamine", "creux_chaud", "creux_froid")
# le nom du produit à imprimer (un message de refus, une ligne de note) : jamais la clé interne
LIBELLES_PRODUIT = {"lamine": "laminé", "creux_chaud": "creux fini à chaud", "creux_froid": "creux formé à froid"}
NUANCES = ("S235", "S275", "S355", "S460")            # les nuances des modules, par limite élastique croissante

# les autres noms que portent des notes enregistrées (bibliothèque des assemblages) : ils suivent la même table
ALIAS: Dict[str, str] = {
    "S355 (fu 490)": "S355",       # ligne reprise d'autres références (f_u = 490) : en Belgique, tableau 3.3 ANB
    "S355 N/NL": "S355N",
    "S460 N/NL": "S460",
}

# ---------------------------------------------------------------- les tableaux de l'ANB (lignes reprises)
# tableau → (norme de produit, {désignation: (bandes R_eH, bandes R_m)}) ; bande = (sup, incluse, valeur)
_I = True    # borne supérieure incluse
_X = False   # borne supérieure exclue
_REH_10025_2 = (16, 40, 63, 80, 100, 150, 200, 250)
_REH_10210 = (16, 40, 63, 80, 100, 120)


def _bandes(sups, valeurs) -> Tuple[Tuple[float, bool, float], ...]:
    return tuple((float(s), _I, float(v)) for s, v in zip(sups, valeurs))


TABLEAUX: Dict[str, Tuple[str, Dict[str, Tuple[tuple, tuple]]]] = {
    "3.3": ("EN 10025-2:2004", {
        "S235JR": (_bandes(_REH_10025_2, (235, 225, 215, 215, 215, 195, 185, 175)),
                   ((3, _X, 360.0), (100, _I, 360.0), (150, _I, 350.0), (250, _I, 340.0))),
        "S275JR": (_bandes(_REH_10025_2, (275, 265, 255, 245, 235, 225, 215, 205)),
                   ((3, _X, 430.0), (100, _I, 410.0), (150, _I, 400.0), (250, _I, 380.0))),
        "S355JR": (_bandes(_REH_10025_2, (355, 345, 335, 325, 315, 295, 285, 275)),
                   ((3, _X, 510.0), (100, _I, 470.0), (150, _I, 450.0), (250, _I, 450.0))),
    }),
    "3.4": ("EN 10025-3:2004", {
        "S355N": (_bandes(_REH_10025_2, (355, 345, 335, 325, 315, 295, 285, 275)),
                  ((100, _I, 470.0), (200, _I, 450.0), (250, _I, 450.0))),
        "S420N": (_bandes(_REH_10025_2, (420, 400, 390, 370, 360, 340, 330, 320)),
                  ((100, _I, 520.0), (200, _I, 500.0), (250, _I, 500.0))),
        "S460N": (_bandes(_REH_10025_2[:7], (460, 440, 430, 410, 400, 380, 370)),
                  ((100, _I, 540.0), (200, _I, 530.0))),
    }),
    # (lot C, C1b) aciers thermomécaniques M/ML : la ligne la plus longue s'arrête à 120 mm (la note d du tableau étend
    # la dernière bande à 150 mm pour les produits longs : non reprise, côté sûr — un refus au-delà de 120 mm)
    "3.5": ("EN 10025-4:2004", {
        "S420M": (_bandes(_REH_10210, (420, 400, 390, 380, 370, 365)),
                  ((40, _I, 520.0), (63, _I, 500.0), (80, _I, 480.0), (100, _I, 470.0), (120, _I, 460.0))),
        "S460M": (_bandes(_REH_10210, (460, 440, 430, 410, 400, 385)),
                  ((40, _I, 540.0), (63, _I, 530.0), (80, _I, 510.0), (100, _I, 500.0), (120, _I, 490.0))),
    }),
    "3.8": ("EN 10210-1:2006", {
        "S235JRH": (_bandes(_REH_10210, (235, 225, 215, 215, 215, 195)),
                    ((3, _I, 360.0), (100, _I, 360.0), (120, _I, 350.0))),
        "S275J0H": (_bandes(_REH_10210, (275, 265, 255, 245, 235, 225)),
                    ((3, _I, 430.0), (100, _I, 410.0), (120, _I, 400.0))),
        "S355J0H": (_bandes(_REH_10210, (355, 345, 335, 325, 315, 295)),
                    ((3, _I, 510.0), (100, _I, 470.0), (120, _I, 450.0))),
    }),
    "3.9": ("EN 10210-1:2006", {
        "S355NH": (_bandes((16, 40, 65), (355, 345, 335)), ((65, _I, 470.0),)),
        "S420NH": (_bandes((16, 40, 65), (420, 400, 390)), ((65, _I, 520.0),)),
        "S460NH": (_bandes((16, 40, 65), (460, 440, 430)), ((65, _I, 540.0),)),
    }),
    "3.10": ("EN 10219-1:2006", {
        "S235JRH": (_bandes((16, 40), (235, 225)), ((3, _X, 360.0), (40, _I, 360.0))),
        "S275J0H": (_bandes((16, 40), (275, 265)), ((3, _X, 430.0), (40, _I, 410.0))),
        "S355J0H": (_bandes((16, 40), (355, 345)), ((3, _X, 510.0), (40, _I, 470.0))),
    }),
    "3.11": ("EN 10219-1:2006", {
        "S355NH": (_bandes((16, 40), (355, 345)), ((40, _I, 470.0),)),
        "S460NH": (_bandes((16, 40), (460, 440)), ((40, _I, 540.0),)),
    }),
    "3.12": ("EN 10219-1:2006", {
        "S420MH": (_bandes((16, 40), (420, 400)), ((40, _I, 500.0),)),
        "S460MH": (_bandes((16, 40), (460, 440)), ((40, _I, 530.0),)),
    }),
}

# nuance du site (après ALIAS) → produit → (tableau, désignation)
LIGNES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "S235": {"lamine": ("3.3", "S235JR"), "creux_chaud": ("3.8", "S235JRH"), "creux_froid": ("3.10", "S235JRH")},
    "S275": {"lamine": ("3.3", "S275JR"), "creux_chaud": ("3.8", "S275J0H"), "creux_froid": ("3.10", "S275J0H")},
    "S355": {"lamine": ("3.3", "S355JR"), "creux_chaud": ("3.8", "S355J0H"), "creux_froid": ("3.10", "S355J0H")},
    "S355N": {"lamine": ("3.4", "S355N"), "creux_chaud": ("3.9", "S355NH"), "creux_froid": ("3.11", "S355NH")},
    "S460": {"lamine": ("3.4", "S460N"), "creux_chaud": ("3.9", "S460NH"), "creux_froid": ("3.11", "S460NH")},
    # (lot C, C1b) les lignes du module Barres seul (acier/barre/nuances.py) : S420 — norme produit N/NL (EN 10025-3, à
    # défaut) — et M/ML (EN 10025-4) de S420 et S460. Un produit sans ligne dans l'ANB (S420 formé à froid N/NL ; un creux
    # fini à chaud M/ML, que l'EN 10210-1 ne prévoit pas) : refus « valeur à fournir ». ``NUANCES`` ne les nomme pas : les
    # autres modules gardent leur liste.
    "S420": {"lamine": ("3.4", "S420N"), "creux_chaud": ("3.9", "S420NH")},
    "S420M": {"lamine": ("3.5", "S420M"), "creux_froid": ("3.12", "S420MH")},
    "S460M": {"lamine": ("3.5", "S460M"), "creux_froid": ("3.12", "S460MH")},
}


class HorsTable(ValueError):
    """Nuance, produit ou épaisseur que les tableaux de l'ANB ne couvrent pas : aucune valeur n'est inventée."""


def _cle(nuance: str) -> str:
    n = ALIAS.get(nuance, nuance)
    if n not in LIGNES:
        raise HorsTable(f"nuance inconnue : {nuance}")
    return n


def _tableau_de(nuance: str, produit: str) -> Tuple[str, str]:
    """(tableau, désignation) de ``nuance`` pour ``produit`` ; HorsTable si l'ANB n'en a pas de ligne."""
    if produit not in PRODUITS:
        raise HorsTable(f"produit inconnu : {produit}")
    lg = LIGNES[_cle(nuance)]
    if produit not in lg:
        raise HorsTable(f"{nuance} {LIBELLES_PRODUIT[produit]} : aucune ligne dans les tableaux de l'ANB (valeur à fournir)")
    return lg[produit]


def ligne(nuance: str, produit: str = "lamine") -> Dict[str, str]:
    """La ligne de l'ANB employée pour ``nuance`` et ``produit`` : {"tableau", "norme", "designation"}."""
    tab, des = _tableau_de(nuance, produit)
    return {"tableau": f"{tab} ANB", "norme": TABLEAUX[tab][0], "designation": des}


def _bandes_de(nuance: str, produit: str, rang: int):
    tab, des = _tableau_de(nuance, produit)
    return TABLEAUX[tab][1][des][rang]


def _lire(bandes, t: float, quoi: str, nuance: str, produit: str) -> Tuple[float, int]:
    if not isinstance(t, (int, float)) or t != t or t <= 0.0:
        raise HorsTable(f"épaisseur invalide : {t!r}")
    for i, (sup, incluse, v) in enumerate(bandes):
        if t < sup or (incluse and t <= sup):
            return v, i
    sup = bandes[-1][0]
    raise HorsTable(f"{nuance} {LIBELLES_PRODUIT[produit]} de {t:g} mm : le tableau "
                    f"{ligne(nuance, produit)['tableau']} s'arrête à {sup:g} mm ({quoi})")


def fy(nuance: str, t: float, produit: str = "lamine") -> float:
    """f_y (= R_eH minimale, N/mm²) pour l'épaisseur nominale ``t`` (mm)."""
    return _lire(_bandes_de(nuance, produit, 0), t, "f_y", nuance, produit)[0]


def fu(nuance: str, t: float, produit: str = "lamine") -> float:
    """f_u (= R_m minimale, N/mm²) pour l'épaisseur nominale ``t`` (mm)."""
    return _lire(_bandes_de(nuance, produit, 1), t, "f_u", nuance, produit)[0]


def _texte_bande(bandes, i: int) -> str:
    sup, incluse, _ = bandes[i]
    signe = "≤" if incluse else "<"
    if i == 0:
        return f"t {signe} {sup:g} mm"
    inf, inf_incluse, _ = bandes[i - 1]
    return f"{inf:g} {'<' if inf_incluse else '≤'} t {signe} {sup:g} mm"


def source(nuance: str, t: float, produit: str = "lamine", grandeur: str = "fy") -> str:
    """La référence à imprimer : « tableau 3.3 ANB (EN 10025-2:2004, S355JR), 16 < t ≤ 40 mm »."""
    bandes = _bandes_de(nuance, produit, 0 if grandeur == "fy" else 1)
    _, i = _lire(bandes, t, grandeur, nuance, produit)
    lg = ligne(nuance, produit)
    return f"tableau {lg['tableau']} ({lg['norme']}, {lg['designation']}), {_texte_bande(bandes, i)}"


def nominale(nuance: str) -> float:
    """Le nombre du nom (S235 → 235) : f_y de la plus petite épaisseur d'un produit laminé. Pour un libellé, un
    contrôle de domaine (« la nuance est-elle une S235 ? ») et la méthode ELS des barres, qui prend la f_y nominale
    comme les classeurs du bureau (brief v12 § 13.2 point 5) — jamais pour une résistance de l'Eurocode."""
    return _bandes_de(nuance, "lamine", 0)[0][2]


def epaisseur_max(nuance: str, produit: str = "lamine") -> float:
    """La plus grande épaisseur que le tableau couvre pour f_y (mm)."""
    return _bandes_de(nuance, produit, 0)[-1][0]


def paliers(nuance: str, produit: str = "lamine", grandeur: str = "fy") -> List[Tuple[str, float]]:
    """[(« 16 < t ≤ 40 mm », valeur)] : la ligne entière, pour un tableau à l'écran."""
    bandes = _bandes_de(nuance, produit, 0 if grandeur == "fy" else 1)
    return [(_texte_bande(bandes, i), v) for i, (_, _, v) in enumerate(bandes)]


def bornes(nuance: str, produit: str = "lamine", grandeur: str = "fy") -> List[Tuple[float, bool, str, float]]:
    """[(borne supérieure en mm, borne incluse, « 16 < t ≤ 40 mm », valeur)] : la ligne entière avec ses bornes
    chiffrées, pour un écran qui range ses colonnes par palier — sans lire la structure interne TABLEAUX."""
    bandes = _bandes_de(nuance, produit, 0 if grandeur == "fy" else 1)
    return [(sup, inc, _texte_bande(bandes, i), v) for i, (sup, inc, v) in enumerate(bandes)]

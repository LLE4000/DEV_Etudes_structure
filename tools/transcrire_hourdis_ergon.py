# -*- coding: utf-8 -*-
"""Transcription du catalogue des hourdis Ergon (Guide technique 2018) — HORS chemin de production.

Produit ``beton/donnees/hourdis/ergon_2018.json`` (la source unique du catalogue dans l'application) et la MESURE
datée ``docs/validation/hourdis_ergon_transcription_<date>.md``.

Trois lectures indépendantes du même document, qui doivent s'accorder :

1. la TRANSCRIPTION à la main, page par page (``TRANSCRIPTION`` ci-dessous) — c'est elle qui est livrée ;
2. la RELECTURE du texte du PDF par PyMuPDF (mots et positions) : caractéristiques (p. 20), tableaux M_Rd / V_Rd,c /
   M_Rd avec couche collaborante, poids à transporter, joints et cotes des coupes (p. 29-35), réservations (p. 24),
   appuis (p. 25), largeurs rétrécies (p. 26), ancrages en alvéole (p. 28). Tout écart arrête le script : une valeur
   livrée a été lue deux fois, par deux moyens ;
3. le RELEVÉ VECTORIEL des coupes (p. 29-35) : le contour gris et les alvéoles blanches, en mm (échelle : la largeur
   1196 mm du tracé), d'où l'aire, l'inertie et les modules recalculés, confrontés aux valeurs publiées (p. 20). Le
   relevé est livré comme DESSIN de la coupe ; les valeurs de calcul restent celles du fabricant.

Les écarts du guide lui-même (une cote mal reportée, deux pages qui ne disent pas la même chose) sont NOMMÉS dans
``meta.ecarts`` : la valeur livrée est celle que les deux autres lectures confirment, l'imprimé est gardé à côté.

Lancement : python3 tools/transcrire_hourdis_ergon.py <ERGON2018_GuideTechnique_FR.pdf>
            (PyMuPDF requis ; il n'entre jamais dans le chemin de production)
"""
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import date

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SORTIE = os.path.join(RACINE, "beton", "donnees", "hourdis", "ergon_2018.json")
LARGEUR_TRACE = 1196.0          # mm : la largeur réelle d'un élément, celle de la cote « 1196 » sous chaque coupe
TOLERANCE_DESSIN = 0.15         # mm : simplification des contours relevés (Douglas-Peucker)

# ================================================================================================ la transcription
META = {
    "version": 1,
    "fabricant": "Ergon",
    "edition": "2018",
    "document": "Ergon · Guide technique 2018 · Éléments de plancher et de toiture (ERGON2018_GuideTechnique_FR)",
    "gamme": "Hourdis alvéolés précontraints SP, SPN et SPZ",
    "unites": {"h, b, cotes, appuis, réservations, largeurs": "mm", "poids_kN_m": "kN/m (avec joints)",
               "poids_kN_m2": "kN/m² (avec joints)", "A": "mm²", "Ix": "mm⁴", "W_sup, W_inf": "mm³",
               "M_Rd, M_Rd_chape": "kN·m par élément", "V_Rd": "kN par élément", "transport": "kg/m",
               "joints": "l/m²", "ancrage_alveole": "kN", "dessin": "mm, origine au coin inférieur gauche, y vers le haut"},
    "largeur_nominale": {"valeur": 1200, "page": 20,
                         "texte": "largeur nominale 1200 mm (module), largeur réelle de l'élément 1196 mm"},
    "beton": {"classe": "C55/67", "fck": 55.0, "pages": [13, 20],
              "texte": "classe de résistance C 55/67 pour les hourdis précontraints SP, SPN et SPZ"},
    # « court » : ce que l'écran affiche en une case ; « texte » : la phrase du guide, pour « Pourquoi ? »
    "classe_consequence": {"court": "CC2", "texte": "catalogue valable pour la classe de conséquence CC2", "page": 8},
    "environnement": {"court": "EI ou XC1", "texte": "conception standard en classe d'environnement EI ou classe "
                                                     "d'exposition XC1 ; autres exigences sur demande", "page": 13},
    "feu": {"court": "Rf 60", "texte": "résistance au feu standard Rf 60 ; Rf 90 ou Rf 120 à la demande", "pages": [13, 20]},
    "ancrage_alveole": {"court": "M8 à M12, simple traction",
                        "texte": "chevilles à expansion ou chimiques dans l'alvéole, M8, M10 ou M12 : charge admissible en "
                                 "simple traction, à réduire si l'ancrage est aussi cisaillé ; vérifier la disposition des "
                                 "torons avant de forer"},
    "valeurs": {"nature": "Rd", "pages": [29, 30, 31, 32, 33, 34, 35],
                "texte": "valeurs de CALCUL à l'état-limite ultime, par élément : M_Ed ≤ M_Rd et V_Ed ≤ V_Rd,c, M_Ed et "
                         "V_Ed calculés avec le poids propre de l'élément et la charge utile multipliés par leurs "
                         "coefficients de pondération"},
    "chape": {"epaisseur_min": 50, "classe": "C30/37", "fck": 30.0, "page": 23,
              "treillis": "treillis minimum 6x6x150x150 DE 500S, à déterminer par le bureau d'étude",
              "texte": "couche collaborante : épaisseur minimale conseillée 50 mm au point le plus haut du hourdis, "
                       "béton C 30/37 ; les tableaux donnent M_Rd « avec 50 mm de couche collaborante », pas V_Rd,c"},
    "joints": {"texte": "remplissage des joints au béton C 30/37, granulats ≤ 8 mm, de préférence aussitôt après "
                        "la pose ; aucune circulation avant durcissement", "page": 25},
    "notation": {"texte": "SP 200-4-2X : épaisseur 200 mm, 4 torons Ø 12,5 mm et 2 torons Ø 9,3 mm (indice X) ; "
                          "indice Y : toron Ø 15,2 mm (SPZ 500)", "pages": [29, 35],
                 "torons": {"": 12.5, "X": 9.3, "Y": 15.2}},
    "non_publie": ["V_Rd,c avec couche collaborante", "moment de fissuration M_cr",
                   "pression locale admissible sur la membrure supérieure",
                   "courbes de performance (charge utile admissible selon la portée, p. 21-22 et 29-35) : graphiques, "
                   "non transcrits"],
}

# p. 20 (caractéristiques), p. 29-35 (une page par profil), p. 24-28 (mise en œuvre)
TRANSCRIPTION = [
    {"nom": "SP 150", "page": 29, "h": 150, "b": 1196, "poids_kN_m": 2.94, "poids_kN_m2": 2.45, "A": 114e3, "Ix": 291e6,
     "W_sup": 3.84e6, "W_inf": 3.93e6, "transport": 281, "joints": 4.9,
     "chaine_horizontale": [101, 142, 142, 142, 142, 142, 142, 142, 101], "chaine_verticale": [75, 75],
     "alveoles": {"nombre": 8, "forme": "circulaire", "diametre": 98},
     "appui": {"beton_acier": 80, "maconnerie": 100},
     "retrecis": [400, 540, 680, 820, 960, 1100],
     "reservations": {"angles": [600, 400], "abouts": [600, 400], "faces_laterales": [1000, 400], "circulaire": 80,
                      "rectangulaire": [1000, 400]},
     "ancrage_alveole": 1.0,
     "variantes": [("SP 150-5X", 44, 95, 61), ("SP 150-6X", 52, 97, 73), ("SP 150-7X", 61, 99, 83),
                   ("SP 150-8X", 69, 100, 93), ("SP 150-9X", 77, 101, 103)]},
    {"nom": "SP 200", "page": 30, "h": 200, "b": 1196, "poids_kN_m": 3.33, "poids_kN_m2": 2.78, "A": 127e3, "Ix": 630e6,
     "W_sup": 6.22e6, "W_inf": 6.38e6, "transport": 314, "joints": 6.8,
     "chaine_horizontale": [128, 188, 188, 188, 188, 188, 128], "chaine_verticale": [100, 100],
     "alveoles": {"nombre": 6, "forme": "circulaire", "diametre": 149},
     "appui": {"beton_acier": 80, "maconnerie": 120},
     "retrecis": [520, 700, 890, 1080],
     "reservations": {"angles": [600, 380], "abouts": [600, 400], "faces_laterales": [1000, 380], "circulaire": 130,
                      "rectangulaire": [1000, 400]},
     "ancrage_alveole": 2.5,
     "variantes": [("SP 200-4X", 51, 89, 66), ("SP 200-5X", 63, 91, 81), ("SP 200-6X", 76, 92, 96),
                   ("SP 200-3-2X", 92, 92, 116), ("SP 200-4-2X", 113, 94, 141), ("SP 200-5-2X", 134, 96, 164)]},
    {"nom": "SPN 200", "page": 31, "h": 200, "b": 1196, "poids_kN_m": 3.51, "poids_kN_m2": 2.92, "A": 134e3, "Ix": 648e6,
     "W_sup": 6.24e6, "W_inf": 6.74e6, "transport": 330, "joints": 6.9,
     "chaine_horizontale": [96, 132, 138, 133, 99, 99, 133, 138, 132, 96], "chaine_verticale": [40, 40, 75, 45],
     "alveoles": {"nombre": 9, "forme": "profilée", "largeurs": [93, 102, 102, 93, 44, 93, 102, 102, 93], "hauteur": 140},
     "appui": {"beton_acier": 80, "maconnerie": 120},
     "retrecis": [370, 500, 600, 700, 830, 960, 1100],
     "reservations": {"angles": [600, 400], "abouts": [600, 400], "faces_laterales": [1000, 400], "circulaire": 90,
                      "rectangulaire": [1000, 400]},
     "ancrage_alveole": None,
     "variantes": [("SPN 200-6x", 76, 101, 97), ("SPN 200-8x", 100, 104, 127), ("SPN 200-10x", 124, 107, 156),
                   ("SPN 200-2-8x", 142, 109, 176), ("SPN 200-4-6x", 160, 111, 196), ("SPN 200-6-4x", 176, 113, 215),
                   ("SPN 200-8-2x", 191, 115, 233)]},
    {"nom": "SP 265", "page": 32, "h": 265, "b": 1196, "poids_kN_m": 4.59, "poids_kN_m2": 3.83, "A": 174e3, "Ix": 1467e6,
     "W_sup": 10.64e6, "W_inf": 11.54e6, "transport": 429, "joints": 9.8,
     "chaine_horizontale": [154, 222, 222, 222, 222, 154], "chaine_verticale": [34, 197, 34],
     "chaine_imprimee": [154, 222, 222, 222, 222, 222],
     "alveoles": {"nombre": 5, "forme": "oblongue", "largeur": 169, "hauteur": 197},
     "appui": {"beton_acier": 80, "maconnerie": 120},
     "retrecis": [600, 830, 1050],
     "reservations": {"angles": [600, 380], "abouts": [600, 400], "faces_laterales": [1000, 380], "circulaire": 130,
                      "rectangulaire": [1000, 400]},
     "ancrage_alveole": 4.0,
     "variantes": [("SP 265-6X", 106, 137, 127), ("SP 265-2-4X", 133, 138, 159), ("SP 265-8X", 141, 140, 167),
                   ("SP 265-4-2X", 160, 140, 189), ("SP 265-10X", 175, 142, 205), ("SP 265-5-2X", 190, 142, 222),
                   ("SP 265-6-2X", 220, 143, 254), ("SP 265-7-2X", 251, 144, 289), ("SP 265-8-2X", 279, 145, 319)]},
    {"nom": "SP 320", "page": 33, "h": 320, "b": 1196, "poids_kN_m": 4.94, "poids_kN_m2": 4.12, "A": 187e3, "Ix": 2421e6,
     "W_sup": 15.03e6, "W_inf": 15.23e6, "transport": 461, "joints": 11.0,
     "chaine_horizontale": [185.5, 275, 275, 275, 185.5], "chaine_verticale": [36, 248, 36],
     "alveoles": {"nombre": 4, "forme": "oblongue", "largeur": 218, "hauteur": 248},
     "appui": {"beton_acier": 130, "maconnerie": 150},
     "retrecis": [750, 1020],
     "reservations": {"angles": [600, 260], "abouts": [600, 400], "faces_laterales": [1000, 260], "circulaire": 150,
                      "rectangulaire": [1000, 400]},
     "ancrage_alveole": 4.0,
     "variantes": [("SP 320-6X", 132, 168, 153), ("SP 320-7X", 154, 171, 178), ("SP 320-8X", 175, 172, 202),
                   ("SP 320-5", 195, 170, 223), ("SP 320-6", 233, 173, 265), ("SP 320-7", 271, 175, 305),
                   ("SP 320-8", 308, 178, 345), ("SP 320-9", 346, 180, 387), ("SP 320-10", 382, 183, 425),
                   ("SP 320-11", 418, 185, 464)]},
    {"nom": "SP 400", "page": 34, "h": 400, "b": 1196, "poids_kN_m": 5.93, "poids_kN_m2": 4.94, "A": 222e3, "Ix": 4541e6,
     "W_sup": 22.44e6, "W_inf": 22.97e6, "transport": 549, "joints": 14.7,
     "chaine_horizontale": [193, 270, 270, 270, 193], "chaine_verticale": [42, 316, 42],
     "alveoles": {"nombre": 4, "forme": "oblongue", "largeur": 216, "hauteur": 316},
     "appui": {"beton_acier": 130, "maconnerie": 150},
     "retrecis": [750, 1020],
     "reservations": {"angles": [600, 260], "abouts": [600, 400], "faces_laterales": [1000, 260], "circulaire": 150,
                      "rectangulaire": [1000, 400]},
     "ancrage_alveole": 6.0,
     "variantes": [("SP 400-5", 251, 223, 280), ("SP 400-6", 300, 225, 332), ("SP 400-7", 349, 227, 384),
                   ("SP 400-8", 398, 229, 436), ("SP 400-9", 447, 229, 487), ("SP 400-10", 495, 231, 539),
                   ("SP 400-11", 543, 232, 589), ("SP 400-12", 589, 233, 639), ("SP 400-13", 635, 234, 687)]},
    {"nom": "SPZ 500", "page": 35, "h": 500, "b": 1196, "poids_kN_m": 7.56, "poids_kN_m2": 6.30, "A": 276e3, "Ix": 8550e6,
     "W_sup": 33.78e6, "W_inf": 34.63e6, "transport": 681, "joints": 18.2, "joints_p20": 16.6,
     "chaine_horizontale": [193, 270, 270, 270, 193], "chaine_verticale": [47, 406, 47],
     "alveoles": {"nombre": 4, "forme": "oblongue", "largeur": 206, "hauteur": 406},
     "appui": {"beton_acier": 130, "maconnerie": 200},
     "retrecis": [750, 1020],
     "reservations": {"angles": [600, 260], "abouts": [600, 400], "faces_laterales": [1000, 260], "circulaire": 150,
                      "rectangulaire": [1000, 400]},
     "ancrage_alveole": None,
     "variantes": [("SPZ 500-8", 497, 342, 544), ("SPZ 500-10", 616, 348, 669), ("SPZ 500-11", 675, 353, 731),
                   ("SPZ 500-13", 784, 370, 845), ("SPZ 500-14", 841, 385, 896), ("SPZ 500-16", 950, 396, 1011),
                   ("SPZ 500-14-2Y", 1006, 394, 1069)]},
]

ECARTS = [
    {"profil": "SP 265", "page": 32, "texte": "la dernière cote de la chaîne horizontale est imprimée 222 ; le bord "
     "vaut 154 (symétrie, somme 1196, relevé du tracé 153,9) : 154 est livré"},
    {"profil": "SP 400", "page": 20, "texte": "la coupe de la page 20 cote les alvéoles 218 et répète 270 à la place "
     "du dernier 193 ; la page 34 cote 216 et 193 (relevé du tracé 215,9 et 193,0) : la page 34 est livrée"},
    {"profil": "SPZ 500", "page": 35, "texte": "remplissage des joints 18,2 l/m² à la page 35, 16,6 l/m² au tableau "
     "de la page 20 : les deux sont livrés ; le poids propre publié (avec joints) n'en dépend pas"},
]

PAGES_MISE_EN_OEUVRE = {"reservations": 24, "appui": 25, "retrecis": 26, "ancrage_alveole": 28, "caracteristiques": 20}


# ================================================================================================ la relecture du PDF
def _nombre(t: str) -> float:
    return float(t.replace(",", "."))


def _lignes(page, tol=2.0):
    """Les mots d'une page groupés en lignes — deux mots sont sur la même ligne si leurs ordonnées médianes diffèrent
    de moins de ``tol`` pt d'un mot au suivant, dans l'ordre vertical —, chaque ligne de gauche à droite."""
    mots = sorted(page.get_text("words"), key=lambda w: (w[1] + w[3]) / 2.0)
    lignes, y_prec = [], None
    for w in mots:
        y = (w[1] + w[3]) / 2.0
        if y_prec is None or y - y_prec >= tol:
            lignes.append([])
        lignes[-1].append(w)
        y_prec = y
    return [sorted(li, key=lambda w: w[0]) for li in lignes]


def relire(doc) -> list:
    """Relit dans le texte du PDF tout ce que la transcription livre ; rend la liste des écarts (vide si tout s'accorde)."""
    ecarts = []

    def ecart(ou, attendu, lu):
        ecarts.append(f"{ou} : transcrit {attendu!r}, lu {lu!r}")

    par_nom = {p["nom"]: p for p in TRANSCRIPTION}
    # p. 20 : une ligne par profil — h, b, kN/m, kN/m², A, Ix, Wsup, Winf, joints, 1/R
    vus = set()
    for li in _lignes(doc[19]):
        mots = [w[4] for w in li]
        txt = " ".join(mots)
        m = re.match(r"^(SPN?Z? \d{3})\s+(.*)$", txt)
        if not m or m.group(1) not in par_nom:
            continue
        nums = [_nombre(x) for x in m.group(2).split() if re.fullmatch(r"[\d,]+", x)]
        if len(nums) < 9:
            continue
        p = par_nom[m.group(1)]
        vus.add(p["nom"])
        attendu = [p["h"], p["b"], p["poids_kN_m"], p["poids_kN_m2"], p["A"] / 1e3, p["Ix"] / 1e6, p["W_sup"] / 1e6,
                   p["W_inf"] / 1e6, p.get("joints_p20", p["joints"])]
        if any(abs(a - b) > 1e-9 for a, b in zip(attendu, nums[:9])):
            ecart(f"p. 20 {p['nom']}", attendu, nums[:9])
    for nom in par_nom:
        if nom not in vus:
            ecart("p. 20", nom, "ligne absente")
    # p. 29-35 : tableau, transport, joints, cotes
    for p in TRANSCRIPTION:
        pg = doc[p["page"] - 1]
        lu = []
        for li in _lignes(pg):
            txt = " ".join(w[4] for w in li)
            m = re.match(r"^(SPN?Z?)\s+(\d{3}-[\dxXyY\-]+)\s+(\d+)\s+(\d+)\s+(\d+)", txt)
            if m:
                lu.append((f"{m.group(1)} {m.group(2)}", int(m.group(3)), int(m.group(4)), int(m.group(5))))
        if lu != [tuple(v) for v in p["variantes"]]:
            ecart(f"p. {p['page']} tableau {p['nom']}", p["variantes"], lu)
        texte = pg.get_text()
        m = re.search(r"Poids à transporter\s*:\s*(\d+)\s*kg/m", texte)
        if not m or int(m.group(1)) != p["transport"]:
            ecart(f"p. {p['page']} poids à transporter", p["transport"], m and m.group(1))
        m = re.search(r"Remplissage des joints\s*:\s*([\d,]+)\s*l", texte)
        if not m or abs(_nombre(m.group(1)) - p["joints"]) > 1e-9:
            ecart(f"p. {p['page']} joints", p["joints"], m and m.group(1))
        # la chaîne horizontale : la ligne de nombres juste au-dessus de « 1196 »
        lignes = _lignes(pg)
        i1196 = next(i for i, li in enumerate(lignes) if [w[4] for w in li] == ["1196"])
        chaine = [_nombre(w[4]) for w in lignes[i1196 - 1]]
        if chaine != list(p.get("chaine_imprimee", p["chaine_horizontale"])):
            ecart(f"p. {p['page']} chaîne horizontale", p.get("chaine_imprimee", p["chaine_horizontale"]), chaine)
        # les autres cotes de la coupe : entre le titre et la chaîne horizontale
        y_titre = min(w[3] for li in lignes for w in li if w[4] == p["nom"].split()[0])
        y_chaine = lignes[i1196 - 1][0][1]
        dans = sorted(_nombre(w[4]) for li in lignes for w in li
                      if y_titre + 1 < w[1] < y_chaine - 1 and re.fullmatch(r"\d+(,\d+)?", w[4]))
        a = p["alveoles"]
        etiquettes = ([a["diametre"]] if "diametre" in a else
                      sorted(set(a["largeurs"])) + [a["hauteur"]] if "largeurs" in a else [a["largeur"]])
        attendu = sorted([p["h"]] + list(p["chaine_verticale"]) + etiquettes)
        if dans != attendu:
            ecart(f"p. {p['page']} cotes de la coupe {p['nom']}", attendu, dans)
    # p. 24 : réservations (L, B des angles, abouts, faces latérales ; Ø circulaire ; L, B rectangulaire)
    t24 = doc[23].get_text().split("\n")
    for p in TRANSCRIPTION:
        i = next((k for k, x in enumerate(t24) if x.strip() == p["nom"]), None)
        vals = [int(x) for x in t24[i + 1:i + 10]] if i is not None else None
        r = p["reservations"]
        attendu = r["angles"] + r["abouts"] + r["faces_laterales"] + [r["circulaire"]] + r["rectangulaire"]
        if vals != attendu:
            ecart(f"p. 24 réservations {p['nom']}", attendu, vals)
    # p. 25 : longueurs d'appui — « a = 80 mm pour SP 150, SP 200, SP 265 et SPN 200 »
    t25 = doc[24].get_text()
    appuis = {}
    for m in re.finditer(r"a = (\d+) mm pour ([^\n]+)", t25):
        for nom in re.findall(r"SPN? ?Z? ?\d{3}", m.group(2)):
            appuis.setdefault(nom.replace("  ", " "), []).append(int(m.group(1)))
    for p in TRANSCRIPTION:
        if appuis.get(p["nom"]) != [p["appui"]["beton_acier"], p["appui"]["maconnerie"]]:
            ecart(f"p. 25 appuis {p['nom']}", p["appui"], appuis.get(p["nom"]))
    # p. 26 : largeurs réalisables des hourdis rétrécis
    t26 = doc[25].get_text().split("\n")
    for p in TRANSCRIPTION:
        i = next((k for k, x in enumerate(t26) if x.strip() == p["nom"]), None)
        vals = [int(x) for x in t26[i + 1].split("/")] if i is not None else None
        if vals != p["retrecis"]:
            ecart(f"p. 26 rétrécis {p['nom']}", p["retrecis"], vals)
    # p. 28 : chevilles à expansion dans l'alvéole (et chimiques : les mêmes valeurs)
    t28 = doc[27].get_text()
    lus = {}
    for nom, v in re.findall(r"(SP \d{3}):\s*([\d,]+) kN", t28):
        lus.setdefault(nom, []).append(_nombre(v))
    for p in TRANSCRIPTION:
        attendu = [p["ancrage_alveole"]] * 2 if p["ancrage_alveole"] is not None else None
        if lus.get(p["nom"]) != attendu:
            ecart(f"p. 28 ancrage en alvéole {p['nom']}", attendu, lus.get(p["nom"]))
    return ecarts


# ================================================================================================ le relevé vectoriel
def _bezier(p0, p1, p2, p3, n=12):
    out = []
    for i in range(1, n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3
        out.append((a * p0.x + b * p1.x + c * p2.x + d * p3.x, a * p0.y + b * p1.y + c * p2.y + d * p3.y))
    return out


def _sous_chemins(dessin):
    """Les sous-chemins d'un tracé PyMuPDF en listes de points (pt)."""
    chemins, cour = [], []

    def debut(x, y):
        nonlocal cour
        if not cour or abs(cour[-1][0] - x) > 1e-3 or abs(cour[-1][1] - y) > 1e-3:
            if cour:
                chemins.append(cour)
            cour = [(x, y)]

    for it in dessin["items"]:
        if it[0] == "l":
            debut(it[1].x, it[1].y)
            cour.append((it[2].x, it[2].y))
        elif it[0] == "c":
            debut(it[1].x, it[1].y)
            cour += _bezier(it[1], it[2], it[3], it[4])
        elif it[0] == "re":
            r = it[1]
            if cour:
                chemins.append(cour)
                cour = []
            chemins.append([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)])
        else:
            raise ValueError(f"élément de tracé inattendu : {it[0]!r}")
    if cour:
        chemins.append(cour)
    return chemins


def _douglas_peucker(pts, tol):
    if len(pts) < 3:
        return list(pts)
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy)
    dmax, imax = -1.0, 0
    for i in range(1, len(pts) - 1):
        x, y = pts[i]
        d = abs(dy * (x - x0) - dx * (y - y0)) / n if n > 1e-12 else math.hypot(x - x0, y - y0)
        if d > dmax:
            dmax, imax = d, i
    if dmax <= tol:
        return [pts[0], pts[-1]]
    return _douglas_peucker(pts[:imax + 1], tol)[:-1] + _douglas_peucker(pts[imax:], tol)


def _simplifier(poly, tol):
    """Un polygone fermé simplifié (le premier point est aussi coupé en deux arcs) ; sans point final répété."""
    pts = list(poly)
    if math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-6:
        pts = pts[:-1]
    k = max(range(len(pts)), key=lambda i: math.hypot(pts[i][0] - pts[0][0], pts[i][1] - pts[0][1]))
    a = _douglas_peucker(pts[:k + 1], tol)
    b = _douglas_peucker(pts[k:] + [pts[0]], tol)
    return a[:-1] + b[:-1]


def proprietes(contour, alveoles):
    """Aire, ordonnée du centre de gravité, inertie et modules d'une section pleine moins ses alvéoles (polygones en mm)."""
    def integrales(pts):
        A = Sx = Ixx = 0.0
        for i in range(len(pts)):
            (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
            c = x0 * y1 - x1 * y0
            A += c
            Sx += (y0 + y1) * c
            Ixx += (y0 * y0 + y0 * y1 + y1 * y1) * c
        s = 1.0 if A > 0 else -1.0
        return s * A / 2.0, s * Sx / 6.0, s * Ixx / 12.0

    A, S, I = integrales(contour)
    for v in alveoles:
        a, s, i = integrales(v)
        A, S, I = A - a, S - s, I - i
    h = max(y for _, y in contour)
    yg = S / A
    Ig = I - A * yg * yg
    return {"A": A, "yg": yg, "Ix": Ig, "W_sup": Ig / (h - yg), "W_inf": Ig / yg, "h": h}


def relever(doc, p) -> dict:
    """Le relevé vectoriel de la coupe d'un profil : le seul remplissage gris dans la page (les gris hors page, laissés
    par la mise en page, sont écartés) et les remplissages blancs qu'il contient ; échelle 1196 mm / largeur du gris."""
    pg = doc[p["page"] - 1]
    tr = pg.get_drawings()
    gris = [d for d in tr if d.get("fill") and abs(d["fill"][0] - d["fill"][1]) < 0.05 and 0.3 < d["fill"][0] < 0.8
            and d["rect"].y1 < 330 and d["rect"].x0 >= 0 and d["rect"].y0 >= 0]
    if len(gris) != 1:
        raise SystemExit(f"{p['nom']} : {len(gris)} remplissages gris dans la page, un seul attendu")
    R = gris[0]["rect"]
    blancs = [d for d in tr if d.get("fill") and min(d["fill"]) > 0.95 and R.contains(d["rect"]) and d["rect"].width < R.width * 0.5]
    e = LARGEUR_TRACE / R.width

    def mm(q):
        return ((q[0] - R.x0) * e, (R.y1 - q[1]) * e)

    contours = [[mm(q) for q in c] for c in _sous_chemins(gris[0])]
    if len(contours) != 1:
        raise SystemExit(f"{p['nom']} : {len(contours)} sous-chemins au contour, un seul attendu")
    alveoles = sorted(([mm(q) for q in c] for b in blancs for c in _sous_chemins(b)), key=lambda v: min(x for x, _ in v))
    brut = proprietes(contours[0], alveoles)
    contour = [(round(x, 1), round(y, 1)) for x, y in _simplifier(contours[0], TOLERANCE_DESSIN)]
    alv = [[(round(x, 1), round(y, 1)) for x, y in _simplifier(v, TOLERANCE_DESSIN)] for v in alveoles]
    livre = proprietes(contour, alv)
    return {"contour": contour, "alveoles": alv, "brut": brut, "livre": livre, "echelle": e}


# ================================================================================================ l'écriture
def _profil_json(p, rel) -> dict:
    out = {"nom": p["nom"], "page": p["page"], "h": p["h"], "b": p["b"], "poids_kN_m": p["poids_kN_m"],
           "poids_kN_m2": p["poids_kN_m2"], "A": round(p["A"]), "Ix": round(p["Ix"]), "W_sup": round(p["W_sup"]),
           "W_inf": round(p["W_inf"]), "transport": p["transport"], "joints": p["joints"]}
    if "joints_p20" in p:
        out["joints_p20"] = p["joints_p20"]
    out.update({"cotes": {"chaine_horizontale": p["chaine_horizontale"], "chaine_verticale": p["chaine_verticale"],
                          "alveoles": p["alveoles"]},
                "appui": p["appui"], "retrecis": p["retrecis"], "reservations": p["reservations"],
                "ancrage_alveole": p["ancrage_alveole"],
                "variantes": [{"nom": n, "M_Rd": m, "V_Rd": v, "M_Rd_chape": mc} for n, m, v, mc in p["variantes"]],
                "dessin": {"source": f"relevé vectoriel de la coupe, p. {p['page']}",
                           "contour": [list(q) for q in rel["contour"]],
                           "alveoles": [[list(q) for q in v] for v in rel["alveoles"]]}})
    return out


def _ecrire_json(obj, niveau=0) -> str:
    """JSON lisible et reproductible : un niveau par ligne, mais une liste de nombres ou de points sur une seule ligne."""
    ind = " " * niveau
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        corps = ",\n".join(f"{ind} {json.dumps(k, ensure_ascii=False)}: {_ecrire_json(v, niveau + 1)}" for k, v in obj.items())
        return "{\n" + corps + "\n" + ind + "}"
    if isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj) or all(
                isinstance(x, list) and all(not isinstance(y, (dict, list)) for y in x) for x in obj):
            return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
        corps = ",\n".join(f"{ind} {_ecrire_json(x, niveau + 1)}" for x in obj)
        return "[\n" + corps + "\n" + ind + "]"
    return json.dumps(obj, ensure_ascii=False)


def _pct(a, b):
    return 100.0 * (a - b) / b


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    import fitz      # PyMuPDF : outil seulement
    doc = fitz.open(sys.argv[1])
    ecarts = relire(doc)
    if ecarts:
        raise SystemExit("La transcription et la relecture du PDF divergent :\n  " + "\n  ".join(ecarts))
    jour = date.today().isoformat()
    releves = {p["nom"]: relever(doc, p) for p in TRANSCRIPTION}
    meta = dict(META, transcription=f"{jour} · tools/transcrire_hourdis_ergon.py · "
                                    f"docs/validation/hourdis_ergon_transcription_{jour}.md",
                pages=PAGES_MISE_EN_OEUVRE, ecarts=ECARTS)
    catalogue = {"meta": meta, "profils": [_profil_json(p, releves[p["nom"]]) for p in TRANSCRIPTION]}
    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    with open(SORTIE, "w", encoding="utf-8") as fh:
        fh.write(_ecrire_json(catalogue) + "\n")
    # la mesure datée
    lignes = [f"# Transcription du catalogue des hourdis Ergon — {jour}", "",
              "*MESURE datée (ne se réécrit pas). Produite par `tools/transcrire_hourdis_ergon.py` depuis le guide "
              "technique Ergon 2018 (92 pages) : trois lectures indépendantes — transcription à la main, relecture du "
              "texte du PDF, relevé vectoriel des coupes — qui s'accordent. Le fichier livré est "
              "`beton/donnees/hourdis/ergon_2018.json`.*", "",
              f"**{len(TRANSCRIPTION)} profils, {sum(len(p['variantes']) for p in TRANSCRIPTION)} variantes "
              "(M_Rd, V_Rd,c, M_Rd avec 50 mm de couche collaborante) ; relecture du PDF : 0 écart.**", "",
              "## 1 · Relecture du texte du PDF", "",
              "Relues mot à mot et comparées à la transcription : les caractéristiques (p. 20 : h, b, poids avec "
              "joints en kN/m et kN/m², A, I_x, W_sup, W_inf, joints), les sept tableaux M_Rd / V_Rd,c / M_Rd avec "
              "couche collaborante (p. 29-35, ordre des lignes compris), le poids à transporter et le remplissage des "
              "joints de chaque page, la chaîne de cotes horizontale et les autres cotes de chaque coupe, les "
              "réservations maximales (p. 24), les longueurs d'appui (p. 25), les largeurs des hourdis rétrécis "
              "(p. 26) et les charges admissibles des chevilles dans l'alvéole (p. 28). **Aucun écart.**", "",
              "## 2 · Relevé vectoriel des coupes, confronté aux valeurs publiées", "",
              "Contour gris et alvéoles blanches de chaque coupe, à l'échelle de la cote 1196 mm ; aire, inertie et "
              "modules intégrés sur le relevé brut, puis sur le dessin livré (simplifié à "
              f"{C_F(TOLERANCE_DESSIN, 2)} mm). Les valeurs de calcul livrées restent celles du fabricant : le relevé "
              "sert au dessin et au contrôle.", "",
              "| Profil | p. | h relevé | alvéoles | A (10³ mm²) | écart | I_x (10⁶ mm⁴) | écart | W_sup | écart | W_inf | écart |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in TRANSCRIPTION:
        r = releves[p["nom"]]["livre"]
        lignes.append(f"| {p['nom']} | {p['page']} | {C_F(r['h'], 1)} | {len(releves[p['nom']]['alveoles'])} | "
                      f"{C_F(r['A'] / 1e3, 1)} ({C_F(p['A'] / 1e3, 0)}) | {C_F(_pct(r['A'], p['A']), 1)} % | "
                      f"{C_F(r['Ix'] / 1e6, 0)} ({C_F(p['Ix'] / 1e6, 0)}) | {C_F(_pct(r['Ix'], p['Ix']), 1)} % | "
                      f"{C_F(r['W_sup'] / 1e6, 2)} ({C_F(p['W_sup'] / 1e6, 2)}) | {C_F(_pct(r['W_sup'], p['W_sup']), 1)} % | "
                      f"{C_F(r['W_inf'] / 1e6, 2)} ({C_F(p['W_inf'] / 1e6, 2)}) | {C_F(_pct(r['W_inf'], p['W_inf']), 1)} % |")
    ecart_profil = {p["nom"]: max(abs(_pct(releves[p["nom"]]["livre"][k], p[k])) for k in ("A", "Ix", "W_sup", "W_inf"))
                    for p in TRANSCRIPTION}
    pire_nom = max(ecart_profil, key=ecart_profil.get)
    pire = ecart_profil[pire_nom]
    autres = max(v for k, v in ecart_profil.items() if k != pire_nom)
    plus_petite = releves[pire_nom]["livre"]["A"] < next(p["A"] for p in TRANSCRIPTION if p["nom"] == pire_nom)
    lignes += ["", f"Plus grand écart : {C_F(pire, 1)} % ({pire_nom} : la section dessinée est un peu plus "
               f"{'petite' if plus_petite else 'grande'} que la section de calcul du fabricant) ; les autres profils à "
               f"{C_F(autres, 1)} % au plus.", "",
               "## 3 · Écarts du guide lui-même", ""]
    lignes += [f"- **{e['profil']}** (p. {e['page']}) : {e['texte']}." for e in ECARTS]
    lignes += ["", "## 4 · Ce que le guide ne publie pas", ""]
    lignes += [f"- {t}" for t in META["non_publie"]]
    os.makedirs(os.path.join(RACINE, "docs", "validation"), exist_ok=True)
    with open(os.path.join(RACINE, "docs", "validation", f"hourdis_ergon_transcription_{jour}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lignes) + "\n")
    print(f"{SORTIE} : {len(TRANSCRIPTION)} profils ; plus grand écart du relevé {pire:.1f} %")


def C_F(x, nd):
    """Un nombre à la française (virgule décimale), sans « -0 »."""
    x = round(x, nd)
    return f"{0.0 if x == 0 else x:.{nd}f}".replace(".", ",")


if __name__ == "__main__":
    main()

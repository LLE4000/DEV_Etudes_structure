# -*- coding: utf-8 -*-
"""Le TABLEAU DES ARMATURES du bureau, recalculé : les valeurs de l'image « Tableau armature.png » (racine du dépôt)
que l'écran Streamlit ``modules/tableau_armatures.py`` affiche — et n'affiche que cela, sans aucun calcul.

L'image porte, pour douze diamètres (lignes) :
- la section de n barres, n = 1 à 10 (mm²) — la même colonne se lit aussi « n barres par mètre », c'est-à-dire un
  espacement de 100 / n cm (l'en-tête porte 100,0 · 50,0 · 33,3 … 10,0), en mm²/m ;
- la masse linéique d'une barre (kg/m, trois décimales), avec ρ = 7 860 kg/m³ (écrit dans l'en-tête) ;
- la section par mètre pour quatre espacements « fixes donnés » : 15, 30, 35 et 40 cm (mm²/m).

Les formules qui reproduisent CHAQUE valeur de l'image (180 nombres, relus dans ses pixels par
tests/test_armatures_moteur.py) :

    a = π·Ø²/4                      section d'une barre (mm²)
    A = n·a                         n barres (mm²) — ou n barres par mètre (mm²/m)
    A/m = a·100/e                   barres espacées de e cm (mm²/m)
    m = ρ·a·10⁻⁶                    masse linéique d'une barre (kg/m), ρ = 7 860 kg/m³

arrondies au mm² le plus proche (la moitié vers le haut : aucune valeur du tableau n'est à moins de 0,003 mm² d'une
moitié, le sens de l'arrondi n'y change donc rien) et la masse à trois décimales.

Bibliothèque standard seule (math) : le module tourne dans le navigateur (Pyodide).
"""
import math
from typing import Dict, List

REFERENCE = "modules/tableau_armatures.py (image « Tableau armature.png »)"
IMAGE = "Tableau armature.png"
LEGENDE_IMAGE = "Tableau des sections d’armatures en mm² et espacement fixe (cm)"
TITRE_ECRAN = "Tableau des sections d’armatures"                     # le titre de l'écran (sans son émoji)

# les lignes et les colonnes de l'image, dans son ordre
DIAMETRES = (6, 8, 10, 12, 14, 16, 20, 25, 28, 30, 32, 40)         # mm
NOMBRES = tuple(range(1, 11))                                       # barres (ou barres par mètre)
ESPACEMENTS_FIXES = (15, 30, 35, 40)                                # cm (« espacement fixe donné »)
MASSE_VOLUMIQUE = 7860                                              # kg/m³ (l'en-tête de l'image ; l'usage est 7 850)
METRE = 100                                                         # cm : la bande de référence des sections par mètre


def aire(d: float) -> float:
    """La section d'une barre de diamètre ``d`` (mm) : π·Ø²/4, en mm²."""
    return math.pi * d * d / 4


def section(n: float, d: float) -> float:
    """La section de ``n`` barres de diamètre ``d`` : n·π·Ø²/4 (mm²) ; n barres par mètre donnent des mm²/m."""
    return n * aire(d)


def section_par_metre(d: float, e: float) -> float:
    """La section par mètre de barres Ø ``d`` (mm) espacées de ``e`` (cm) : π·Ø²/4 · 100 / e (mm²/m)."""
    return aire(d) * METRE / e


def espacement(n: float) -> float:
    """L'espacement (cm) de ``n`` barres par mètre : 100 / n — la seconde ligne de l'en-tête de l'image."""
    return METRE / n


def masse_lineique(d: float) -> float:
    """La masse d'un mètre de barre Ø ``d`` (mm) : ρ·π·Ø²/4·10⁻⁶ (kg/m), ρ = 7 860 kg/m³ comme l'image."""
    return MASSE_VOLUMIQUE * aire(d) * 1e-6


def arrondi(x: float) -> int:
    """Un nombre du tableau, au mm² le plus proche (la moitié vers le haut, comme un tableur)."""
    return int(math.floor(x + 0.5))


def masse_txt(m: float) -> str:
    """La masse écrite comme dans l'image : trois décimales, virgule (« 0,889 »)."""
    return f"{m:.3f}".replace(".", ",")


def espacement_txt(e: float) -> str:
    """L'espacement d'en-tête écrit comme dans l'image : une décimale, virgule (« 33,3 »)."""
    return f"{e:.1f}".replace(".", ",")


def tableau() -> Dict:
    """Le tableau complet, ligne par ligne, avec les valeurs EXACTES et celles que l'image affiche.

    {"diametres", "nombres", "espacements_n" (100/n, cm), "espacements_fixes" (cm), "masse_volumique",
     "lignes": [{"d", "aire", "masse", "masse_txt",
                 "barres": [n·a pour n = 1…10], "barres_arrondi": [...],
                 "fixes": [a·100/e pour e = 15, 30, 35, 40], "fixes_arrondi": [...]}, ...]}"""
    lignes: List[Dict] = []
    for d in DIAMETRES:
        barres = [section(n, d) for n in NOMBRES]
        fixes = [section_par_metre(d, e) for e in ESPACEMENTS_FIXES]
        m = masse_lineique(d)
        lignes.append({"d": d, "aire": aire(d), "masse": m, "masse_txt": masse_txt(m),
                       "barres": barres, "barres_arrondi": [arrondi(x) for x in barres],
                       "fixes": fixes, "fixes_arrondi": [arrondi(x) for x in fixes]})
    return {"reference": REFERENCE, "diametres": list(DIAMETRES), "nombres": list(NOMBRES),
            "espacements_n": [espacement(n) for n in NOMBRES], "espacements_n_txt": [espacement_txt(espacement(n)) for n in NOMBRES],
            "espacements_fixes": list(ESPACEMENTS_FIXES), "masse_volumique": MASSE_VOLUMIQUE, "lignes": lignes}

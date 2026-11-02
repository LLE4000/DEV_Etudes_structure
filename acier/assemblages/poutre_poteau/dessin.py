# -*- coding: utf-8 -*-
"""Les TRAITS que les trois attaches poutre–poteau ont en commun — une seule
source (consolidation du 02/11/2026, `docs/assemblages/POUTRE_POTEAU.md`
§10.7).

Quatre briques étaient écrites à l'identique dans les trois ``schemas.py`` de
la famille : la MARGE du tronçon de poteau dessiné, les deux NOMS de pièce du
cartouche et des étiquettes, et la SECTION du poteau tournée selon le cas
d'appui. Elles ne sont pas dans le kit de dessin parce qu'elles n'existent
pas chez les grands frères poutre–poutre : un poteau continue au-dessus et en
dessous de l'attache, et c'est le cas d'appui qui décide de son orientation.

Le KIT de dessin lui-même (primitives, ``Feuille`` de cotation, palette,
rendu SVG, lignes de rupture, hachures, ``_section_I``) reste importé de
``poutre_poutre.doubles_cornieres.schemas`` : le déplacer dans un fichier
propre est un CHANTIER à part (§10.5 de l'étude, `docs/CHANTIERS.md`), parce
qu'il oblige à toucher les trois ``schemas.py`` des grands frères, dont les
dessins sont figés par ``tests/test_assemblages_schemas.py``.

PURETÉ : le kit et ``acier.js`` / ``acier.bibliotheques``, rien d'autre —
aucun Streamlit, aucun moteur.
"""
from acier.js import mx
from acier.bibliotheques import PERSO
from acier.assemblages.poutre_poutre.doubles_cornieres.schemas import _section_I


def marge_poteau(R):
    """Hauteur du tronçon de poteau dessiné au-dessus et en dessous de la
    poutre portée (il continue : bords rompus)."""
    return mx(60.0, 0.18 * R.h_S)


def nom_poteau(R):
    """Le nom du POTEAU dans les étiquettes : son profilé, ou « Poteau »
    quand il est saisi dimension par dimension."""
    return "Poteau" if R.u.prof_P == PERSO else R.u.prof_P


def nom_portee(R):
    """Le nom de la POUTRE PORTÉE dans les étiquettes : son profilé, ou
    « Poutre portée » quand elle est saisie dimension par dimension."""
    return "Poutre portée" if R.u.prof_S == PERSO else R.u.prof_S


def section_poteau(R):
    """Contour du poteau en COUPE HORIZONTALE, orienté pour la vue en plan :
    l'axe x du dessin suit l'axe de la poutre portée.

    - cas (a) : x = X (profondeur hP), y = Y (largeur d'aile bP) — les deux
      ailes sont VERTICALES, aux abscisses x = ±hP/2 ;
    - cas (b) : x = Y (largeur d'aile bP), y = X (profondeur hP) — les deux
      ailes sont HORIZONTALES, aux ordonnées y = ±hP/2, et la poutre entre
      ENTRE elles."""
    sec = _section_I(R.b_P, R.h_P, R.tw_P, R.tf_P, R.r_P)
    if R.ame:
        return [(sx, sy - R.h_P / 2) for sx, sy in sec]
    return [(sy - R.h_P / 2, sx) for sx, sy in sec]

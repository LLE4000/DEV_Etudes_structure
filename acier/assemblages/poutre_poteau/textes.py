# -*- coding: utf-8 -*-
"""Les MOTS que les trois attaches poutre–poteau ont en commun — une seule
source (consolidation du 02/11/2026, `docs/assemblages/POUTRE_POTEAU.md`).

Deux familles de textes vivaient en TROIS copies identiques, une par module :

- les **limites de validité** du §8.4 de l'étude, que l'onglet Méthode écrit
  en liste et que l'onglet Vérifications reprend mot pour mot dans le repli
  « Périmètre de la vérification » ;
- la **phrase de recoupe** de l'alerte ``aile_poutre`` et le **message** de
  l'alerte ``ame_hp``, qui disent la même chose aux trois modules.

Ce qui reste PROPRE à chaque module n'est pas remonté ici, et c'est
volontaire — une copie tenue est une vue, une rédaction différente n'est pas
une copie :

- ``LIMITE_AME`` (la septième limite, cas (b) seulement) : le plat d'âme cite
  ``qx = 3·MS/hp²`` parce que son groupe est EXCENTRÉ, la platine d'about ne
  le cite pas parce que le sien est CENTRÉ, et les cornières parlent de la
  traction des boulons. Chaque module écrit la sienne et la passe à
  ``limites()`` ;
- l'alerte ``b1_long`` : elle n'existe QUE pour le plat d'âme (c'est le bras
  ``z`` du plat qui triple dans la variante b1) ;
- la ``w`` chiffrée de ``aile_poutre`` au-delà de la phrase commune : le plat
  d'âme ajoute « ou éloigner la poutre (gh ≥ …) », les deux autres non.

PURETÉ : aucune importation. Des chaînes et deux fonctions qui les composent.
"""

# =========================================================================
#  1. Les limites de validité — étude §8.4
# =========================================================================
# Les neuf limites : les six premières et les deux dernières valent dans les
# deux cas d'appui ; la septième (``LIMITE_AME``, propre à chaque module) ne
# vaut qu'en cas (b) et se passe à ``limites()``.
LIMITES_AVANT = (
    "panneau d'âme du poteau en cisaillement sous les moments d'étage (§6.2.6.1) — une attache simple "
    "n'y contribue pas, il se vérifie avec l'ensemble des attaches de l'étage",
    "poteau sous charges combinées : résistance de section, flambement, déversement, et le moment "
    "VEd·e que cette attache lui apporte (EC3-1-1 §6.2 et §6.3)",
    "torsion du poteau induite par une attache d'un seul côté",
    "attaches des DEUX côtés du poteau : efforts opposés, boulons ou âme partagés — pression "
    "diamétrale, cisaillement local et tying non traités",
    "capacité de ROTATION au-delà des règles de détail : l'articulation est supposée (§5.2), le "
    "contact de la semelle inférieure de la poutre sur le poteau n'est pas vérifié",
    "cisaillement global, flexion et déversement de la poutre portée",
)
LIMITES_APRES = (
    "poteau raidi (goussets, raidisseurs entre ailes) : les longueurs efficaces du Tab. 6.4 et les "
    "limites de l'âme supposent le poteau NON raidi",
    "ELS : flèches, vibrations, confort",
)


def limites(limite_ame, ame=True):
    """Les NEUF limites du §8.4 pour le cas d'appui courant.

    ``limite_ame`` : la septième, propre au module (flexion hors plan de
    l'âme du poteau — chaque attache la rédige selon ce qui sollicite cette
    âme chez elle) ; ``ame`` : vrai en cas (b), seul cas où elle s'applique.
    """
    return LIMITES_AVANT + ((limite_ame,) if ame else ()) + LIMITES_APRES


# =========================================================================
#  2. Les alertes communes aux trois modules
# =========================================================================
# ``aile_poutre`` : la semelle de la poutre portée ne passe pas entre les
# ailes du poteau et la recoupe est insuffisante (cas b, bloquante).
M_AILE_POUTRE = ("Configuration impossible : la semelle de la poutre portée ne passe pas entre "
                 "les ailes du poteau et la recoupe est insuffisante.")


def w_aile_poutre(bS, libre_ailes, formule_about, rec_min, rec_actuelle):
    """La phrase CHIFFRÉE de l'alerte ``aile_poutre``, commune aux trois
    modules. ``formule_about`` nomme ce qui fixe la position de l'about :
    « gh » pour un plat d'âme ou des cornières, « tp » pour une platine
    d'about (l'about est alors à l'épaisseur de la platine de la face
    d'appui). Les quatre autres arguments sont déjà FORMATÉS par l'appelant
    (``acier.formats``) : ce fichier compose, il ne formate pas."""
    return ("bS = " + bS + " mm dépasse la largeur libre entre ailes hP − 2·tfP − 2·j = "
            + libre_ailes + " mm : il faut recouper les semelles sur au moins "
            "(bP − twP)/2 + j − " + formule_about + " = " + rec_min
            + " mm depuis l'about (recoupe actuelle " + rec_actuelle + " mm)")


# ``ame_hp`` : appui dans l'âme, flexion hors plan non couverte (information).
def m_ame_hp(sollicitation, remede, suite=""):
    """Le message de l'alerte ``ame_hp``, commun aux trois modules.

    ``sollicitation`` : ce qui sollicite l'âme hors de son plan chez ce
    module (le groupe excentré du plat d'âme y ajoute ``qx = 3·MS/hp²``, le
    groupe centré de la platine non) ; ``remede`` : ce qui raidirait ;
    ``suite`` : une phrase propre au module, ajoutée telle quelle."""
    return ("Appui dans l'ÂME du poteau : la flexion HORS PLAN de l'âme sous " + sollicitation
            + " n'est PAS couverte par l'EN 1993-1-8 pour une âme non raidie — à justifier "
            "séparément, ou raidir (" + remede + ")." + (" " + suite if suite else ""))

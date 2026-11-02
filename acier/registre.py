# -*- coding: utf-8 -*-
"""Registre déclaratif des assemblages métalliques : famille → assemblages.

Ajouter un assemblage = ajouter un dossier sous ``acier/assemblages/<famille>/``
et une entrée ici. La page de sélection (``modules/assemblages.py``) lit ce
registre ; elle n'est jamais modifiée.

Chaque entrée porte : identifiant, titre, description courte, vignette (petit
SVG propre à l'assemblage), état (« disponible » / « à venir ») et le point
d'entrée de l'interface (module + fonction ``show``).
"""
import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Assemblage:
    id: str
    titre: str
    pitch: str
    vignette: str
    etat: str = "disponible"          # « disponible » | « à venir »
    module: str = ""                  # chemin pointé du module d'interface
    entree: str = "show"
    court: str = ""                   # nom court (≤ 18) du sélecteur de la barre d'outils ; ``titre``
                                      # reste celui du catalogue, du sommaire et des signets du rapport

    @property
    def disponible(self):
        return self.etat == "disponible" and bool(self.module)


@dataclass(frozen=True)
class Famille:
    id: str
    titre: str
    pitch: str
    assemblages: tuple


# ------------------------------------------------------------------ vignettes
_INK, _ACC, _KO = "#15181F", "#33415C", "#9C3341"


def vignette_doubles_cornieres():
    """Élévation schématique : âme porteuse, cornière, boulons."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 90" width="100%" role="img" '
        'aria-label="Doubles cornières d\'âme">'
        f'<rect x="8" y="4" width="46" height="6" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
        f'<rect x="8" y="80" width="46" height="6" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
        f'<rect x="28" y="10" width="6" height="70" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
        f'<polygon points="38,22 38,68 112,68 112,14 50,14 50,22" fill="#E6ECF0" stroke="{_INK}" stroke-width="1"/>'
        f'<rect x="34" y="26" width="30" height="40" fill="{_ACC}" fill-opacity=".28" stroke="{_ACC}" stroke-width="1.2"/>'
        f'<rect x="34" y="26" width="4" height="40" fill="{_ACC}" fill-opacity=".6"/>'
        f'<circle cx="52" cy="34" r="3" fill="#fff" stroke="{_INK}" stroke-width="1"/>'
        f'<circle cx="52" cy="46" r="3" fill="#fff" stroke="{_INK}" stroke-width="1"/>'
        f'<circle cx="52" cy="58" r="3" fill="#fff" stroke="{_INK}" stroke-width="1"/>'
        f'<path d="M22 34H42M22 46H42M22 58H42" stroke="{_KO}" stroke-width="1.6" stroke-dasharray="3 2" fill="none"/>'
        '</svg>')


def vignette_plat_ame():
    """Élévation schématique : plat soudé sur l'âme porteuse, boulonné à
    l'âme de la portée."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 90" width="100%" role="img" '
        'aria-label="Plat d\'âme soudé (fin plate)">'
        f'<rect x="8" y="4" width="46" height="6" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
        f'<rect x="8" y="80" width="46" height="6" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
        f'<rect x="28" y="10" width="6" height="70" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
        f'<polygon points="40,22 40,68 112,68 112,14 52,14 52,22" fill="#E6ECF0" stroke="{_INK}" stroke-width="1"/>'
        f'<rect x="34" y="26" width="34" height="40" fill="{_ACC}" fill-opacity=".28" stroke="{_ACC}" stroke-width="1.2"/>'
        f'<rect x="34" y="26" width="3.5" height="40" fill="#D98A00"/>'
        f'<circle cx="52" cy="34" r="3" fill="#fff" stroke="{_INK}" stroke-width="1"/>'
        f'<circle cx="52" cy="46" r="3" fill="#fff" stroke="{_INK}" stroke-width="1"/>'
        f'<circle cx="52" cy="58" r="3" fill="#fff" stroke="{_INK}" stroke-width="1"/>'
        '</svg>')


def vignette_platine_about():
    """Platine d'about : plat soudé en about de la portée, boulonné sur
    l'âme porteuse (élévation : platine par la tranche, boulons de profil)."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 90" width="100%" role="img" '
        'aria-label="Platine d\'about">'
        f'<rect x="8" y="4" width="46" height="6" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
        f'<rect x="8" y="80" width="46" height="6" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
        f'<rect x="28" y="10" width="6" height="70" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
        f'<polygon points="40,22 40,68 112,68 112,14 52,14 52,22" fill="#E6ECF0" stroke="{_INK}" stroke-width="1"/>'
        f'<rect x="34" y="20" width="6" height="52" fill="{_ACC}" fill-opacity=".35" stroke="{_ACC}" stroke-width="1.2"/>'
        f'<path d="M22 32H46M22 46H46M22 60H46" stroke="{_KO}" stroke-width="1.6" stroke-dasharray="3 2" fill="none"/>'
        '</svg>')


# ----------------------------------------------- vignettes poutre – poteau
# Les trois attaches sur POTEAU sont vues EN PLAN (coupe horizontale au
# niveau des boulons) : c'est la vue qui porte l'information neuve
# (docs/assemblages/POUTRE_POTEAU.md §10.7). Le poteau y est un profil en I
# coupé — deux ailes VERTICALES, qui touchent les bords du cadre pour dire
# qu'il continue au-dessus et en dessous, et une âme horizontale. La poutre
# portée arrive par la droite sur la face extérieure de l'aile : on voit sa
# semelle de dessus (teinte claire) et son âme coupée. Les boulons sont dans
# le plan du dessin : ils se tracent en axes tiretés, comme chez les grands
# frères. Règle de teinte de ces trois vues : ce qui est COUPÉ est en
# #C3CED7 (les ailes et l'âme du poteau, l'âme de la poutre portée), ce qui
# est vu de DESSUS en #E6ECF0 (la semelle supérieure de la poutre portée).
_POTEAU = (f'<rect x="12" y="0" width="7" height="90" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
           f'<rect x="40" y="0" width="7" height="90" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
           f'<rect x="19" y="41" width="21" height="8" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>')


def _poutre_portee(x):
    """Poutre portée vue de dessus depuis ``x`` : semelle puis âme coupée."""
    return (f'<rect x="{x}" y="20" width="{114 - x}" height="50" fill="#E6ECF0" '
            f'stroke="{_INK}" stroke-width="1"/>'
            f'<rect x="{x}" y="41" width="{114 - x}" height="8" fill="#C3CED7" '
            f'stroke="{_INK}" stroke-width="1"/>')


def vignette_plat_ame_poteau():
    """Vue en plan : plat soudé sur l'aile du poteau, au droit de son âme,
    boulonné à l'âme de la poutre portée (cordon en orange)."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 90" width="100%" role="img" '
        'aria-label="Plat d\'âme soudé sur poteau, vue en plan">'
        + _POTEAU + _poutre_portee(54)
        + f'<rect x="47" y="32" width="41" height="9" fill="{_ACC}" fill-opacity=".28" '
          f'stroke="{_ACC}" stroke-width="1.2"/>'
          f'<rect x="47" y="32" width="3.5" height="9" fill="#D98A00"/>'
          f'<path d="M66 27V56M78 27V56" stroke="{_KO}" stroke-width="1.6" '
          f'stroke-dasharray="3 2" fill="none"/>'
        '</svg>')


def vignette_doubles_cornieres_poteau():
    """Vue en plan : deux cornières de part et d'autre de l'âme portée,
    ailes A boulonnées sur l'aile du poteau (deux files à ± p3/2)."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 90" width="100%" role="img" '
        'aria-label="Doubles cornières d\'âme sur poteau, vue en plan">'
        + _POTEAU + _poutre_portee(50)
        + f'<polygon points="47,12 55,12 55,33 83,33 83,41 47,41" fill="{_ACC}" '
          f'fill-opacity=".28" stroke="{_ACC}" stroke-width="1.2"/>'
          f'<polygon points="47,78 55,78 55,57 83,57 83,49 47,49" fill="{_ACC}" '
          f'fill-opacity=".28" stroke="{_ACC}" stroke-width="1.2"/>'
          f'<rect x="47" y="12" width="8" height="21" fill="{_ACC}" fill-opacity=".6"/>'
          f'<rect x="47" y="57" width="8" height="21" fill="{_ACC}" fill-opacity=".6"/>'
          f'<path d="M33 22H62M33 68H62" stroke="{_KO}" stroke-width="1.6" '
          f'stroke-dasharray="3 2" fill="none"/>'
          f'<path d="M72 28V62" stroke="{_KO}" stroke-width="1.6" '
          f'stroke-dasharray="3 2" fill="none"/>'
        '</svg>')


def vignette_platine_about_poteau():
    """Vue en plan : platine d'about par la tranche contre l'aile du poteau,
    soudée sur l'âme portée, boulonnée par deux files symétriques."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 90" width="100%" role="img" '
        'aria-label="Platine d\'about sur poteau, vue en plan">'
        + _POTEAU + _poutre_portee(57)
        + f'<rect x="47" y="18" width="10" height="54" fill="{_ACC}" fill-opacity=".35" '
          f'stroke="{_ACC}" stroke-width="1.2"/>'
          f'<rect x="57" y="37" width="4" height="4" fill="#D98A00"/>'
          f'<rect x="57" y="49" width="4" height="4" fill="#D98A00"/>'
          f'<path d="M33 30H66M33 60H66" stroke="{_KO}" stroke-width="1.6" '
          f'stroke-dasharray="3 2" fill="none"/>'
        '</svg>')


def vignette_pied_poteau():
    """Pied de poteau : poteau, platine, mortier, tiges d'ancrage dans le massif
    (élévation schématique)."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 90" width="100%" role="img" '
        'aria-label="Pied de poteau sur béton">'
        f'<rect x="10" y="56" width="100" height="30" fill="#E6ECF0" stroke="{_INK}" stroke-width="1"/>'
        f'<rect x="28" y="52" width="64" height="4" fill="#D9D4C7" stroke="{_INK}" stroke-width="0.6"/>'
        f'<rect x="28" y="46" width="64" height="6" fill="{_ACC}" fill-opacity=".35" stroke="{_ACC}" stroke-width="1.2"/>'
        f'<rect x="54" y="6" width="12" height="40" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
        f'<rect x="44" y="6" width="32" height="4" fill="#C3CED7" stroke="{_INK}" stroke-width="1"/>'
        f'<path d="M38 46V78M82 46V78" stroke="{_KO}" stroke-width="2" fill="none"/>'
        f'<rect x="34" y="76" width="8" height="4" fill="{_KO}"/><rect x="78" y="76" width="8" height="4" fill="{_KO}"/>'
        f'<path d="M36 40h4M80 40h4" stroke="{_INK}" stroke-width="1"/>'
        '</svg>')


# ------------------------------------------------------------------- registre
FAMILLES = (
    Famille(
        id="poutre_poutre",
        titre="Poutre – Poutre",
        pitch="Attaches d'une poutre portée sur une poutre porteuse",
        assemblages=(
            Assemblage(
                id="doubles_cornieres",
                titre="Doubles cornières d'âme",
                court="Doubles cornières",
                pitch="Assemblage articulé : deux cornières boulonnées ou soudées sur l'âme "
                      "de la poutre portée et sur l'âme de la porteuse — EN 1993-1-8.",
                vignette=vignette_doubles_cornieres(),
                etat="disponible",
                module="acier.assemblages.poutre_poutre.doubles_cornieres.interface",
            ),
            Assemblage(
                id="plat_ame",
                titre="Plat d'âme soudé (fin plate)",
                court="Plat d'âme",
                pitch="Assemblage articulé : plat vertical soudé sur l'âme de la porteuse, "
                      "boulonné à l'âme de la poutre portée — EN 1993-1-8, ANB belge.",
                vignette=vignette_plat_ame(),
                etat="disponible",
                module="acier.assemblages.poutre_poutre.plat_ame.interface",
            ),
            Assemblage(
                id="platine_about",
                titre="Platine d'about (hauteur partielle)",
                court="Platine d'about",
                pitch="Assemblage articulé : platine mince soudée en about de la poutre portée, "
                      "boulonnée sur l'âme de la porteuse (2 files) — EN 1993-1-8, ANB belge, tying.",
                vignette=vignette_platine_about(),
                etat="disponible",
                module="acier.assemblages.poutre_poutre.platine_about.interface",
            ),
        ),
    ),
    Famille(
        id="poutre_poteau",
        titre="Poutre – Poteau",
        pitch="Attaches d'une poutre portée sur l'aile ou dans l'âme d'un poteau",
        assemblages=(
            Assemblage(
                id="plat_ame_poteau",
                titre="Plat d'âme soudé sur poteau (fin plate)",
                court="Plat d'âme/poteau",
                pitch="Assemblage articulé : plat vertical soudé sur l'aile du poteau ou dans son âme, "
                      "boulonné à l'âme de la poutre portée — EN 1993-1-8, ANB belge, tying.",
                vignette=vignette_plat_ame_poteau(),
                etat="disponible",
                module="acier.assemblages.poutre_poteau.plat_ame_poteau.interface",
            ),
            Assemblage(
                id="doubles_cornieres_poteau",
                titre="Doubles cornières d'âme sur poteau",
                court="Cornières/poteau",
                pitch="Assemblage articulé : deux cornières boulonnées ou soudées sur l'aile du poteau "
                      "ou dans son âme, et sur l'âme de la poutre portée — EN 1993-1-8, tying.",
                vignette=vignette_doubles_cornieres_poteau(),
                etat="à venir",
                module="acier.assemblages.poutre_poteau.doubles_cornieres_poteau.interface",
            ),
            Assemblage(
                id="platine_about_poteau",
                titre="Platine d'about (hauteur partielle) sur poteau",
                court="Platine/poteau",
                pitch="Assemblage articulé : platine mince soudée en about de la poutre portée, boulonnée "
                      "sur l'aile du poteau ou dans son âme (2 files) — EN 1993-1-8, ANB belge, tying.",
                vignette=vignette_platine_about_poteau(),
                etat="à venir",
                module="acier.assemblages.poutre_poteau.platine_about_poteau.interface",
            ),
        ),
    ),
    Famille(
        id="pied_poteau",
        titre="Pied de poteau",
        pitch="Platine de pied de poteau sur béton : N + Mx + My, tiges d'ancrage, cordons",
        assemblages=(
            Assemblage(
                id="platine_beton",
                titre="Platine de pied de poteau sur béton",
                court="Pied de poteau",
                pitch="Calcul manuel de la platine (boulons libres, éléments soudés, raidisseurs : bras de levier, "
                      "moment pris, épaisseur requise) ; vérification complète EN 1993-1-8 + EN 1992-4 pour une "
                      "grille régulière — ANB belge.",
                vignette=vignette_pied_poteau(),
                etat="disponible",
                module="acier.assemblages.pied_poteau.platine_beton.interface",
            ),
        ),
    ),
)


def trouver(identifiant):
    """``(famille, assemblage)`` pour un identifiant d'assemblage, sinon None."""
    for f in FAMILLES:
        for a in f.assemblages:
            if a.id == identifiant:
                return f, a
    return None


def disponibles():
    return [a for f in FAMILLES for a in f.assemblages if a.disponible]


def charger(assemblage):
    """Le module d'interface de l'assemblage (import paresseux)."""
    return importlib.import_module(assemblage.module)

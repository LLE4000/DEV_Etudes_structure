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
    """Platine d'about (à venir) : plat soudé en about, boulonné sur l'âme."""
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
                pitch="Assemblage articulé : deux cornières boulonnées ou soudées sur l'âme "
                      "de la poutre portée et sur l'âme de la porteuse — EN 1993-1-8.",
                vignette=vignette_doubles_cornieres(),
                etat="disponible",
                module="acier.assemblages.poutre_poutre.doubles_cornieres.interface",
            ),
            Assemblage(
                id="plat_ame",
                titre="Plat d'âme soudé (fin plate)",
                pitch="Assemblage articulé : plat vertical soudé sur l'âme de la porteuse, "
                      "boulonné à l'âme de la poutre portée — EN 1993-1-8, ANB belge.",
                vignette=vignette_plat_ame(),
                etat="disponible",
                module="acier.assemblages.poutre_poutre.plat_ame.interface",
            ),
            Assemblage(
                id="platine_about",
                titre="Platine d'about",
                pitch="Plat soudé en about de la poutre portée, boulonné sur l'âme de la porteuse.",
                vignette=vignette_platine_about(),
                etat="à venir",
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

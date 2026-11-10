# -*- coding: utf-8 -*-
"""Les modules du site, trouvés sans liste à tenir : un fichier api_web/<cle>.py qui déclare ``MODULE = "<cle>"``
EST un module du site (son nom de fichier et sa clé sont la même chose). Il déclare aussi ``PAGE = "<adresse>"``,
l'adresse de son établi (« garde-corps » → garde-corps.html, servie sans « .html »).

Le routeur (le navigateur), le générateur du site (pages, accueil, js/modules.js, paquet du moteur) et les tests lisent
ce registre : ajouter un module, c'est ajouter son fichier — aucun fichier partagé à modifier.

La clé est celle des familles de l'accueil (site/maquettes/construire.py, FAMILLES) : le nom affiché, l'icône et la
famille viennent de là (une seule source).
"""
import importlib
import os
import re

ICI = os.path.dirname(os.path.abspath(__file__))
_MODULE = re.compile(r'^MODULE\s*=\s*"([a-z_]+)"\s*$', re.M)
_PAGE = re.compile(r'^PAGE\s*=\s*"([a-z0-9-]+)"\s*$', re.M)


def _lire(fichier: str) -> str:
    with open(os.path.join(ICI, fichier), encoding="utf-8") as f:
        return f.read()


def cles() -> list:
    """Les clés des modules présents, lues dans les fichiers SANS les importer (le navigateur n'importe que celui
    qu'il ouvre). Un fichier qui déclare MODULE sans PAGE n'est pas (encore) un module."""
    out = []
    for f in sorted(os.listdir(ICI)):
        if f.endswith(".py") and not f.startswith("_"):
            src = _lire(f)
            m = _MODULE.search(src)
            if m and m.group(1) == f[:-3] and _PAGE.search(src):
                out.append(m.group(1))
    return out


def pages() -> dict:
    """{clé: adresse de l'établi} — « garde_corps » → « garde-corps »."""
    out = {}
    for cle in cles():
        m = _PAGE.search(_lire(cle + ".py"))
        if not m:
            raise ValueError(f"api_web/{cle}.py : PAGE = \"…\" manquant")
        out[cle] = m.group(1)
    if len(set(out.values())) != len(out):
        raise ValueError(f"deux modules ont la même adresse : {out}")
    return out


_CLE = re.compile(r"[a-z_]+")
_MODULES: dict = {}   # les modules déjà reconnus et importés (le moteur du navigateur sert TOUS les écrans d'une session)


def module(cle: str):
    """Le module d'une clé, importé à la demande ; une clé inconnue (ou un fichier qui n'est pas un module : commun,
    routeur…) est refusée.

    PARESSEUX (REFONTE_ETABLI.md, S4-D9) : on ne lit QUE le fichier ``<clé>.py``, une fois — la règle est celle de
    ``cles()`` (fichier présent, nom qui ne commence pas par « _ », ``MODULE = "<clé>"`` et ``PAGE`` déclarés), sans
    relire tout le dossier à chaque requête (≈ 5 ms par requête mesurés, 54 fichiers lus)."""
    if isinstance(cle, str) and cle in _MODULES:
        return _MODULES[cle]
    if not isinstance(cle, str) or not _CLE.fullmatch(cle) or cle.startswith("_") \
            or not os.path.isfile(os.path.join(ICI, cle + ".py")):
        raise KeyError(f"module inconnu : {cle!r}")
    src = _lire(cle + ".py")
    m = _MODULE.search(src)
    if not (m and m.group(1) == cle and _PAGE.search(src)):
        raise KeyError(f"module inconnu : {cle!r}")
    mod = importlib.import_module(f"api_web.{cle}")
    _MODULES[cle] = mod
    return mod

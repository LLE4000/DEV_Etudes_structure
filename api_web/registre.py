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


def module(cle: str):
    """Le module d'une clé, importé à la demande ; une clé inconnue (ou un fichier qui n'est pas un module : commun,
    routeur…) est refusée."""
    if not isinstance(cle, str) or cle not in cles():
        raise KeyError(f"module inconnu : {cle!r}")
    return importlib.import_module(f"api_web.{cle}")

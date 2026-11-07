# -*- coding: utf-8 -*-
"""Les tableaux d'ANCRAGE et de RECOUVREMENT du bureau : ils viendront de l'ingénieur. Ce module ne contient AUCUNE
valeur ; il lit le fichier ``beton/donnees/ancrages_recouvrements.json`` (vide tant que le bureau n'a pas fourni ses
tableaux), le CONTRÔLE et le rend à l'écran, qui l'affiche tel quel — un tableau ajouté au fichier apparaît sur le site
sans une ligne de code.

Le fichier ::

    {"version": 1,
     "titre": "Longueurs d'ancrage et de recouvrement du bureau",
     "etat": "Tableaux du bureau à venir",          le texte montré tant que « tableaux » est vide
     "aide": "…",                                    pour qui remplit le fichier
     "tableaux": [<tableau>, …]}

Un tableau ::

    {"cle": "ancrage_b500",                          identifiant court (lettres, chiffres, « _ », « - »), unique
     "titre": "Ancrage droit · B500",                le titre affiché
     "grandeur": "l_bd",                             le symbole de la grandeur (l_bd, l_0, l_b,rqd…), indices après « _ »
     "unite": "cm",                                  « mm », « cm », « m », ou « Ø » (une longueur en diamètres)
     "source": "Tableau du bureau, révision du …",   d'où vient le tableau (obligatoire)
     "conditions": ["Bonne adhérence", …],           facultatif : les hypothèses (béton, acier, adhérence, forme…)
     "colonnes": {"titre": "Ø", "unite": "mm", "valeurs": [8, 10, 12, 16, 20, 25]},   l'en-tête (nombres ou textes)
     "lignes": [{"libelle": "C25/30", "valeurs": [nombre ou null, …]}, …],          une valeur par colonne
     "notes": ["…"]}                                 facultatif

Règle : une clé manquante, une valeur qui n'est pas un nombre (ou ``null`` : case vide), une ligne qui n'a pas autant
de valeurs que de colonnes, une unité inconnue, deux tableaux de même clé lèvent ``ValueError`` avec l'endroit fautif —
le site montre alors l'erreur, jamais un tableau tronqué.

Bibliothèque standard seule (json, os, re) : le module tourne dans le navigateur (Pyodide).
"""
import copy
import json
import math
import os
import re

CHEMIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ancrages_recouvrements.json")
FICHIER = "beton/donnees/ancrages_recouvrements.json"
VERSION = 1
UNITES = ("mm", "cm", "m", "Ø")
CLES_FICHIER = ("version", "titre", "etat", "aide", "tableaux")
CLES_TABLEAU = ("cle", "titre", "grandeur", "unite", "source", "conditions", "colonnes", "lignes", "notes")
_CLE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _texte(x, ou: str, vide: bool = False) -> str:
    if not isinstance(x, str) or (not vide and not x.strip()):
        raise ValueError(f"{ou} : un texte{'' if vide else ' non vide'} est attendu (reçu {x!r})"[:200])
    return x.strip()


def _textes(x, ou: str) -> list:
    if x is None:
        return []
    if not isinstance(x, list):
        raise ValueError(f"{ou} : une liste de textes est attendue")
    return [_texte(t, f"{ou} n° {i + 1}") for i, t in enumerate(x)]


def _nombre(x, ou: str):
    """Une case : un nombre fini, ou ``null`` (case vide)."""
    if x is None:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{ou} : un nombre (ou null pour une case vide) est attendu (reçu {x!r})"[:200])
    return x


def _tableau(t, i: int) -> dict:
    ou = f"tableau n° {i + 1}"
    if not isinstance(t, dict):
        raise ValueError(f"{ou} : un objet est attendu")
    inconnues = sorted(str(k) for k in t if k not in CLES_TABLEAU)
    if inconnues:
        raise ValueError(f"{ou} : clé(s) inconnue(s) {', '.join(inconnues)} — clés admises : {', '.join(CLES_TABLEAU)}")
    cle = _texte(t.get("cle"), f"{ou}, cle")
    if not _CLE.match(cle):
        raise ValueError(f"{ou}, cle : lettres, chiffres, « _ » ou « - » seulement, 40 au plus (reçu {cle!r})")
    ou = f"tableau « {cle} »"
    unite = t.get("unite")
    if unite not in UNITES:
        raise ValueError(f"{ou}, unite : {unite!r} n'est pas permise ({' · '.join(UNITES)})")
    col = t.get("colonnes")
    if not isinstance(col, dict) or not isinstance(col.get("valeurs"), list) or not col["valeurs"]:
        raise ValueError(f"{ou}, colonnes : {{\"titre\", \"unite\", \"valeurs\": [...]}} attendu, une valeur au moins")
    entetes = []
    for j, v in enumerate(col["valeurs"]):
        if isinstance(v, str):
            entetes.append(_texte(v, f"{ou}, colonne n° {j + 1}"))
        else:
            n = _nombre(v, f"{ou}, colonne n° {j + 1}")
            if n is None:
                raise ValueError(f"{ou}, colonne n° {j + 1} : un en-tête ne peut pas être vide")
            entetes.append(n)
    lignes = t.get("lignes")
    if not isinstance(lignes, list) or not lignes:
        raise ValueError(f"{ou}, lignes : une liste d'une ligne au moins est attendue")
    out_l = []
    for k, li in enumerate(lignes):
        oul = f"{ou}, ligne n° {k + 1}"
        if not isinstance(li, dict) or set(li) - {"libelle", "valeurs"}:
            raise ValueError(f"{oul} : {{\"libelle\", \"valeurs\"}} attendu")
        vals = li.get("valeurs")
        if not isinstance(vals, list) or len(vals) != len(entetes):
            raise ValueError(f"{oul} : {len(entetes)} valeur(s) attendue(s), une par colonne (reçu "
                             f"{len(vals) if isinstance(vals, list) else repr(vals)})")
        out_l.append({"libelle": _texte(li.get("libelle"), f"{oul}, libelle"),
                      "valeurs": [_nombre(v, f"{oul}, colonne n° {j + 1}") for j, v in enumerate(vals)]})
    return {"cle": cle, "titre": _texte(t.get("titre"), f"{ou}, titre"), "grandeur": _texte(t.get("grandeur"), f"{ou}, grandeur"),
            "unite": unite, "source": _texte(t.get("source"), f"{ou}, source"),
            "conditions": _textes(t.get("conditions"), f"{ou}, conditions"),
            "colonnes": {"titre": _texte(col.get("titre", ""), f"{ou}, colonnes.titre", vide=True),
                         "unite": _texte(col.get("unite", ""), f"{ou}, colonnes.unite", vide=True), "valeurs": entetes},
            "lignes": out_l, "notes": _textes(t.get("notes"), f"{ou}, notes")}


def valider(donnees) -> dict:
    """Le contenu du fichier, contrôlé et complété ; ``ValueError`` (en français, avec l'endroit fautif) sinon."""
    if not isinstance(donnees, dict):
        raise ValueError(f"{FICHIER} : un objet JSON est attendu")
    inconnues = sorted(str(k) for k in donnees if k not in CLES_FICHIER)
    if inconnues:
        raise ValueError(f"{FICHIER} : clé(s) inconnue(s) {', '.join(inconnues)}")
    if donnees.get("version") != VERSION:
        raise ValueError(f"{FICHIER} : version {donnees.get('version')!r} inconnue (attendue : {VERSION})")
    tabs = donnees.get("tableaux")
    if not isinstance(tabs, list):
        raise ValueError(f"{FICHIER} : « tableaux » doit être une liste (vide tant que le bureau n'a rien fourni)")
    out = [_tableau(t, i) for i, t in enumerate(tabs)]
    cles = [t["cle"] for t in out]
    doubles = sorted({c for c in cles if cles.count(c) > 1})
    if doubles:
        raise ValueError(f"{FICHIER} : deux tableaux ont la même clé ({', '.join(doubles)})")
    return {"version": VERSION, "titre": _texte(donnees.get("titre"), f"{FICHIER}, titre"),
            "etat": _texte(donnees.get("etat"), f"{FICHIER}, etat"), "aide": _texte(donnees.get("aide", ""), f"{FICHIER}, aide", vide=True),
            "tableaux": out}


def lire(chemin: str = "") -> dict:
    """Le fichier du bureau (``CHEMIN``, lu à l'appel), lu et contrôlé (``valider``). Relu à chaque appel : il est
    petit, et un tableau ajouté se voit sans redémarrer."""
    with open(chemin or CHEMIN, encoding="utf-8") as f:
        return valider(json.load(f))


def tableau(donnees: dict, cle: str):
    """Le tableau de clé ``cle`` (une copie), ou None."""
    t = next((t for t in donnees["tableaux"] if t["cle"] == cle), None)
    return copy.deepcopy(t) if t is not None else None

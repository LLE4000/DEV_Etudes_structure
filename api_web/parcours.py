# -*- coding: utf-8 -*-
"""Le PARCOURS d'un module : toutes les requêtes que l'écran peut envoyer, rejouées sans navigateur.

Partant d'une note neuve, on lit le cadre Données que le module rend et on essaie, chacun sur une copie de la note,
chaque choix offert (chaque option de chaque menu, chaque segment, chaque case, chaque bouton « ajouter »), puis un
second niveau sur les choix qui apparaissent après un premier (un type qui révèle d'autres champs) ; on ouvre chaque
sélecteur et on y choisit une autre ligne ; on ajoute, copie et supprime un élément. Un module peut ajouter ses propres
requêtes : ``PARCOURS = [{"action": "modifier", "chemin": [...], "valeur": ...}, …]`` (appliquées à la note neuve).

Deux usages :
- le générateur du site rejoue ce parcours pour relever les fichiers du moteur à empaqueter (ce qu'un chemin rare
  importe doit être dans le paquet, sinon le navigateur échouerait sur ce chemin) ;
- tests/test_api_web_modules.py exige qu'AUCUNE de ces requêtes n'échoue.
"""
import copy
import html
import json
import re

_SELECT = re.compile(r'<select data-chemin="([^"]+)" data-sorte="choix"[^>]*>(.*?)</select>', re.S)
_OPTION = re.compile(r'<option value="([^"]*)"')
_SEGMENT = re.compile(r'<button type="button" class="([^"]*)" data-chemin="([^"]+)" data-sorte="choix" data-valeur="([^"]*)"')
_CASE = re.compile(r'<input type="checkbox" data-chemin="([^"]+)" data-sorte="case"( checked)?')
_AJOUT = re.compile(r'<button type="button" class="addrow" data-chemin="([^"]+)" data-valeur="([^"]*)"')
_PK = re.compile(r'data-pk="([^"]+)"')
_CHOIX = re.compile(r'class="pk-row(?! cur)[^"]*" data-choix="([^"]+)"')

MAX_REQUETES = 220


def _txt(x: str) -> str:
    return html.unescape(x)


def choix_offerts(donnees: str) -> list:
    """Les changements que le cadre Données propose : [(chemin, valeur, libellé), …], sans la valeur déjà retenue."""
    out = []
    for chemin, opts in _SELECT.findall(donnees):
        sel = re.search(r'<option value="([^"]*)" selected', opts)
        for v in _OPTION.findall(opts):
            if not sel or v != sel.group(1):
                out.append((_txt(chemin), _txt(v), f"menu {_txt(chemin)} = {_txt(v)}"))
    for classe, chemin, v in _SEGMENT.findall(donnees):
        if "on" not in classe.split():
            out.append((_txt(chemin), _txt(v), f"segment {_txt(chemin)} = {_txt(v)}"))
    for chemin, coche in _CASE.findall(donnees):
        out.append((_txt(chemin), not coche, f"case {_txt(chemin)} → {not coche}"))
    for chemin, v in _AJOUT.findall(donnees):
        out.append((_txt(chemin), _txt(v), f"ajout {_txt(chemin)}"))
    return out


def parcourir(cle: str, traiter, niveaux: int = 2, maximum: int = MAX_REQUETES) -> list:
    """Rejoue le parcours d'un module à travers ``traiter`` (le routeur) ; rend [(requête décrite, réponse ok,
    erreur), …]. S'arrête à ``maximum`` requêtes (le compte est rendu, rien n'est tu)."""
    journal = []

    def q(desc, **k):
        if len(journal) >= maximum:
            return None
        r = json.loads(traiter(json.dumps(dict(module=cle, **k))))
        journal.append((desc, bool(r.get("ok")), r.get("erreur")))
        return r if r.get("ok") else None

    base = q("nouveau", action="nouveau")
    if not base:
        return journal
    e0 = base["etat"]
    q("calculer", action="calculer", etat=e0)
    vus = set()
    file = [(e0, base, 1)]
    while file:
        etat, rep, niveau = file.pop(0)
        for chemin, valeur, desc in choix_offerts(rep.get("donnees", "")):
            if (chemin, json.dumps(valeur)) in vus:
                continue
            vus.add((chemin, json.dumps(valeur)))
            r = q(desc, action="modifier", etat=copy.deepcopy(etat), chemin=chemin.split("/"), valeur=valeur)
            if r and niveau < niveaux:
                file.append((r["etat"], r, niveau + 1))
    # les sélecteurs : les ouvrir, y choisir une autre ligne
    for role in dict.fromkeys(_PK.findall(base.get("resultat", ""))):
        s = q(f"sélecteur {role}", action="selecteur", etat=copy.deepcopy(e0), role=role)
        if s:
            autres = _CHOIX.findall(s.get("html", ""))
            if autres:
                q(f"choisir {role} = {_txt(autres[0])}", action="choisir", etat=copy.deepcopy(e0), role=role, nom=_txt(autres[0]))
    # les éléments de la note
    r = q("ajouter un élément", action="ajouter", etat=copy.deepcopy(e0))
    if r:
        r = q("copier l'élément ouvert", action="dupliquer", etat=r["etat"])
        if r:
            q("supprimer le premier élément", action="supprimer", etat=r["etat"], index=0)
    # les requêtes propres au module
    try:
        from . import registre
        propres = getattr(registre.module(cle), "PARCOURS", [])
    except Exception:  # noqa: BLE001 — un module qui ne s'importe pas a déjà échoué plus haut
        propres = []
    for i, req in enumerate(propres):
        req = dict(req)
        req.setdefault("etat", copy.deepcopy(e0))
        q(f"parcours propre n° {i + 1} ({req.get('action')})", **req)
    if len(journal) >= maximum:
        journal.append((f"parcours arrêté à {maximum} requêtes", True, None))
    return journal

# -*- coding: utf-8 -*-
"""La NOTE DE CALCUL PDF du module « Poutres bois » sur le site : la PLOMBERIE entre la note du site et le générateur
``bois/poutre/note.py`` (24/11/2026).

Le générateur fait TOUTE la note en un document : la page de garde du site (projet, partie, date, indice, sommaire —
la garde des notes Poutre, Dalle et Cornières), puis une planche A4 paysage par poutre (deux quand elle se poursuit),
sur l'état de la note du site ({"version": 2, "poutres": [...], "actif": 0}) — qui EST l'entrée du moteur : la note
refait le calcul de chaque poutre par ``moteur.verifier``, la fonction de l'écran (parité prouvée par
tests/test_note_bois.py : les taux, efforts, contraintes et flèches de la note sont ceux du moteur).

L'IDENTIFICATION de la garde et des en-têtes est TOUJOURS celle de la note du site (R2 : une seule source — son nom,
son chantier, son indice, la date du jour que la page envoie), avec la correspondance de la note de la poutre
(api_web/note_poutre.py) : projet = le chantier, à défaut le nom de la note ; partie = le nom de la note quand il y a un
chantier, sinon le titre du module ; indice = celui de la note, « A » à défaut ; date = celle de la page, à défaut celle
du jour (le générateur). Le fichier : « Note de calcul - <nom de la note>.pdf ».

``api_web/bois.telecharger(etat, "note_pdf", infos)`` appelle ``fichier`` ; le routeur contrôle le fichier rendu
(commun.refus_fichier). Une poutre hors domaine, ou une planche qui ne tiendrait pas même poursuivie, est REFUSÉE avec
le message du générateur (ValueError) : jamais une note tronquée ni un chiffre faux.
"""
import base64
import copy
from typing import Dict, Optional, Tuple

TITRE = "Poutres bois"
# les fichiers du dépôt que la NOTE charge en plus de l'écran du module et des paquets : AUCUN — le générateur
# (bois/poutre/note.py) est dans le paquet du MOTEUR (``bois`` est un paquet empaqueté entier, construire.PAQUETS_ENTIERS),
# les seuils des taux et l'arrondi (acier/garde_corps/membres.py, acier/js.py) y sont déjà par l'écran, et le kit (ndc_pdf)
# est tout le paquet de la note. Prouvé par le relevé « comme dans le navigateur » de tests/test_note_bois.py.
FICHIERS: Tuple[str, ...] = ()


def _texte(x) -> str:
    return str(x or "").strip()


def nom_note(etat: Dict, infos: Optional[Dict]) -> str:
    """Le nom de la note : celui que la note du site porte ; à défaut, le nom de la poutre quand elle est seule, le titre
    du module sinon."""
    nom = _texte((infos or {}).get("nom"))
    if nom:
        return nom
    ps = etat.get("poutres") or []
    if len(ps) == 1 and isinstance(ps[0], dict):
        return _texte(ps[0].get("nom")) or TITRE
    return TITRE


def infos_doc(etat: Dict, infos: Optional[Dict]) -> Dict:
    """Les « infos » du générateur ({"nom_projet", "partie", "indice", "date"}) depuis celles de la note du site."""
    infos = infos or {}
    nom, chantier = nom_note(etat, infos), _texte(infos.get("chantier"))
    out = {"nom_projet": chantier or nom, "partie": nom if chantier else TITRE, "indice": _texte(infos.get("indice")) or "A"}
    if _texte(infos.get("date")):
        out["date"] = _texte(infos.get("date"))
    return out


def pdf(etat: Dict, infos: Optional[Dict] = None) -> bytes:
    """La note PDF (octets) de toute la note : la garde, puis la planche de chaque poutre, dans l'ordre de la liste."""
    from bois.poutre import note as NT
    return NT.generer_note(etat, infos_doc(etat, infos))


def fichier(etat: Dict, infos: Optional[Dict] = None) -> Dict:
    """Le fichier à télécharger : {"nom", "type": "application/pdf", "base64"} (le routeur le contrôle) — le même
    contrat que api_web/note_poutre.fichier."""
    from . import commun as C
    infos = dict(infos or {})
    octets = pdf(copy.deepcopy(etat), infos)
    return {"nom": C.nom_fichier(f"Note de calcul - {nom_note(etat, infos)}", "pdf"), "type": "application/pdf",
            "base64": base64.b64encode(octets).decode("ascii")}

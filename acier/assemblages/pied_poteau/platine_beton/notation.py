# -*- coding: utf-8 -*-
"""Textes du pied de poteau pour l'écran et la note : références courtes et
lignes d'alerte. Le moteur écrit en notation finale (fjd, c, Aeff, hef…) ;
les références courtes sont propres à ce module parce qu'il cite trois
normes (EN 1993-1-8, EN 1992-1-1, EN 1992-4)."""
import re

from acier.formats import F

LEGENDE_REFERENCES = ("Références : EC3 = EN 1993-1-8 sauf indication (§, Tab.) ; EC2 = EN 1992-1-1 ; "
                      "EC2-4 = EN 1992-4 (ancrages dans le béton) ; ANB = annexe nationale belge.")

# remplacements CIBLÉS avant le nettoyage des parenthèses (les références
# longues du moteur portent une explication ; la ligne de tableau la perd)
_CIBLES = [
    ("Théorème statique (borne inférieure) sur les composants d'EN 1993-1-8 §6.2.5 et §6.2.4 — modèle "
     "d'ingénieur, non codifié", "théorème statique · §6.2.5, §6.2.4 · modèle d'ingénieur"),
    ("(modèle de la Table 6.7) + interaction linéaire (convexité)", "Tab. 6.7 · interaction linéaire"),
    ("— répartition élastique (EN 1992-4 section 6)", "· EC2-4 §6 (répartition)"),
    ("longueurs efficaces dérivées des Tableaux 6.4 / 6.6", "leff : Tab. 6.4 / 6.6"),
    ("tiges de coin / latérales : longueur plafonnée (modèle d'ingénieur)", "coin / lat. : plafond, modèle d'ingénieur"),
    ("(Fc,fc,Rd = Mc,Rd/(h − tf)), par analogie", "par analogie"),
    ("(la bande c est élastique par construction)", ""),
]
_REMPLACEMENTS = [
    ("EN 1993-1-8 ", ""), ("EN 1992-1-1 ", "EC2 "), ("EN 1992-4", "EC2-4"), ("Tableau ", "Tab. "),
]


def ref_courte(ref):
    """La référence courte d'une vérification (une ligne de tableau)."""
    txt = ref or ""
    for a, b in _CIBLES:
        txt = txt.replace(a, b)
    parts = []
    for part in txt.split(" ; "):
        part = re.sub(r"\s*\((?!répartition)[^()]*\)", "", part).strip()
        for a, b in _REMPLACEMENTS:
            part = part.replace(a, b)
        part = re.sub(r"\s+", " ", part).strip(" ·")
        if part:
            parts.append(part)
    vus = []
    for p in parts:
        if p not in vus:
            vus.append(p)
    return " · ".join(vus)


TITRES_ALERTES = {
    "profil": "Impossible — dimensions du profilé incomplètes",
    "mortier": "Mortier hors conditions de βj = 2/3 (EC3 §6.2.5(7))",
    "massif": "Impossible — la platine déborde du massif",
    "files": "Plus de deux files : rangées traitées tige par tige",
    "dist_e": "Tab. 3.3 — pince des tiges au bord de la platine",
    "dedans": "Impossible — tige dans le vide du profil creux",
    "coin": "Tiges de coin : longueur efficace plafonnée (modèle d'ingénieur)",
    "lat": "Tiges latérales : appui réduit à l'aile (modèle d'ingénieur)",
    "int": "Tiges entre les ailes : tronçon en T non raidi (Tab. 6.4)",
    "eq": "Impossible — équilibre élastique introuvable",
    "soul": "Soulèvement : tiges tendues",
    "comp_dom": "N > 2·FC,Rd : niveau 1 hors domaine, le niveau 2 renseigne",
    "fend": "Bord du massif à moins de 1,5·hef : fendage et éclatement à vérifier (EC2-4 §7.2.1.7-8)",
    "rig": "Rigidité du pied (information)",
    "ancr_info": "Béton d'ancrage (EN 1992-4) non satisfait — hors périmètre, en information",
    "eta_info": "Tige post-installée : valeurs de l'ETA à reporter du document (ψg,Np = 1)",
    "c_ecret": "Bande c écrêtée par la platine : niveau 1 conservatif (platine épaisse)",
    # calcul manuel
    "dist_p": "Tab. 3.3 — entraxe de deux boulons inférieur à 2,2·d0",
    "aucun_e": "Impossible — aucun élément soudé",
    "aucun_b": "Impossible — aucun boulon",
    "plein": "Appui sur toute la platine : approximation entre les ailes et au-delà des abouts",
    "exc": "Efforts excentrés : N·e ajouté aux moments",
    "tige_info": "Un boulon dépasse sa résistance en traction (information, hors verdict)",
    "fjd_info": "Pression sous la platine supérieure à fjd (information, hors verdict)",
}
_TITRES_PREFIXES = [("elem_", "Impossible — élément soudé mal dimensionné"), ("hors_", "Impossible — élément hors de la platine"),
                    ("raidh_", "Impossible — raidisseur hors de la platine"), ("raid_", "Impossible — raidisseur sans longueur ou épaisseur"),
                    ("bin_", "Impossible — boulon dans l'emprise d'un élément"), ("brd_", "Impossible — boulon dans un raidisseur")]


def titre_alerte(a):
    if a.id in TITRES_ALERTES:
        return TITRES_ALERTES[a.id]
    for pre, titre in _TITRES_PREFIXES:
        if a.id.startswith(pre):
            return titre
    return a.msg


def _premiere_phrase(txt):
    m = re.match(r"(.+?\.)(?:\s|$)", txt)
    return (m.group(1) if m else txt).rstrip(".")


def ligne_alerte(a, R):
    """``(titre, détail)`` — le détail est l'explication chiffrée (``why``),
    sinon la ligne du Tableau 3.3, sinon la première phrase du message pour
    les alertes qui ne font qu'informer (soulèvement, rigidité)."""
    if a.why:
        return titre_alerte(a), _premiere_phrase(a.why)
    if a.id == "dist_e":
        for x in R.dist:
            if x.alert == a.id:
                return titre_alerte(a), F(x.val, 1) + " mm < " + F(x.kmin, 1) + "·d0 = " + F(x.min, 1) + " mm"
    if a.id in ("soul", "rig"):
        return titre_alerte(a), _premiere_phrase(a.msg)
    return titre_alerte(a), ""

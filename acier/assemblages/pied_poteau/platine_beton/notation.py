# -*- coding: utf-8 -*-
"""Textes du pied de poteau pour l'écran et la note : références courtes et
lignes d'alerte. Le moteur écrit en notation finale (fjd, c, Aeff, hef…) ;
les références courtes sont propres à ce module parce qu'il cite trois
normes (EN 1993-1-8, EN 1992-1-1, EN 1992-4)."""
import re

from acier.formats import F

# La ligne « Notations » du pied de la note (vérification complète) ET le paragraphe « Notations » de l'onglet
# Méthode : une source (consolidation du 01/11/2026, point 4 — les cotes ne portent plus que leur symbole : bp,
# tg, hef, dh… ; leur définition vit ici). Elle tient sur la ligne de la note (≤ 798 pt à 5,6 pt, contrôlé).
NOTATIONS = ("Notations : bp, lp, tp : platine ; wx, wy : entraxes des files et des rangées ; hef, dh : "
             "ancrage ; tg : mortier ; bc, lc, hc : massif ; a : gorge des cordons ; c : bande efficace "
             "(EC3 §6.2.5) ; fjd : résistance d'appui ; C, pmax : résultante et pression de la répartition "
             "; T : effort d'une tige.")
# Le calcul manuel : sa propre ligne (boulons, bras m, sections au nu), même règle.
NOTATIONS_MANUEL = ("Notations : bp, lp, tp : platine ; (x ; y) : repère au centre ; m : bras jusqu'à l'appui "
                    "(cordon a déduit) ; leff : longueur efficace (tronçon en T) ; b : section au nu (L + "
                    "2·m) ; c : bande efficace ; fjd : résistance d'appui ; C, pmax : résultante, pression ; "
                    "T : effort d'un boulon ; mEd : moment de la bande ; hef, dh : ancrage ; tg : mortier ; "
                    "bc, lc, hc : massif.")

def rangee_lisible(cle):
    """La clé de rangée que le moteur NOMME (« y-:-200 » : axe, côté, coordonnée)
    dite en clair : « y− (y = -200 mm) » — écran (Vérifications, Répartition)
    et note. L'écran et la note affichaient la clé brute (« y-:-200 (coin) »,
    relecture des captures et des notes du 01/11/2026)."""
    try:
        axe, cote, coord = cle[0], cle[1], float(cle.split(":", 1)[1])
    except (IndexError, TypeError, ValueError):
        return cle
    return f"{axe}{'+' if cote == '+' else '−'} ({axe} = {F(coord, 0)} mm)"


LEGENDE_REFERENCES = ("Références : EC3 = EN 1993-1-8 sauf indication (§, Tab.) ; EC2 = EN 1992-1-1 ; "
                      "EC2-4 = EN 1992-4 (ancrages dans le béton) ; ANB = annexe nationale belge.")

# remplacements CIBLÉS avant le nettoyage des parenthèses (les références
# longues du moteur portent une explication ; la ligne de tableau la perd)
_CIBLES = [
    ("Théorème statique (borne inférieure) sur les composants d'EN 1993-1-8 §6.2.5 et §6.2.4 — modèle "
     "d'ingénieur, non codifié", "théorème statique · §6.2.5, §6.2.4 · modèle d'ingénieur"),
    ("(modèle de la Table 6.7) + interaction linéaire (convexité)", "Tab. 6.7 · interaction linéaire"),
    ("— répartition élastique (EN 1992-4 section 6)", "· EC2-4 §6 (répartition)"),
    # la référence du calcul manuel (« … §6.7 — répartition élastique ») : « répartition élastique » est une
    # phrase interdite des notes depuis le 21/09/2026 (test_assemblages_note) — la VUE l'abrège, le moteur
    # garde sa référence (consolidation du 01/11/2026, point 23)
    ("— répartition élastique", "(répartition)"),
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

# -*- coding: utf-8 -*-
"""Écriture des textes du moteur pour l'écran et la note.

Le moteur (``moteur.py``) garde le vocabulaire de l'outil de référence — ses
chaînes font partie de la parité (``alertes[].message``, ``vals``, export
texte). L'écran et la note les AFFICHENT en notation de l'Eurocode et avec
des références courtes ; c'est une vue, pas une seconde vérité : une seule
fonction (``ec``) porte la correspondance, et un test la garde.

- ``ec(txt)``          : « Lc » → « hc », « ln » → « c », « dnt » → « dc,sup »,
                         « dnb » → « dc,inf », « dn » → « dc », « décalage » → « Δz » ;
- ``ref_courte(ref)``  : « EN 1993-1-8 Tableau 3.4 ; groupe excentré : MSB
                         Part 5 §4.2.1.1 » → « Tab. 3.4 · MSB §4.2.1.1 »
                         (EN 1993-1-8 est implicite : légende en pied de page) ;
- ``titre_alerte(a)`` / ``ligne_alerte(a, R)`` : la ligne courte d'une alerte,
                         composée des données de l'alerte (jamais recalculée).
"""
import re

from acier.formats import F

# --- notation ---------------------------------------------------------
_REGLES = [
    (re.compile(r"(?<![A-Za-z])Lc(?![A-Za-z])"), "hc"),
    (re.compile(r"(?<![A-Za-z])ln(?![A-Za-z])"), "c"),
    (re.compile(r"(?<![A-Za-z])dnt(?![A-Za-z])"), "dc,sup"),
    (re.compile(r"(?<![A-Za-z])dnb(?![A-Za-z])"), "dc,inf"),
    (re.compile(r"(?<![A-Za-z])dn(?![A-Za-z])"), "dc"),
    (re.compile(r"(?<![A-Za-z])décalage(?![A-Za-z])"), "Δz"),
]

LEGENDE_NOTATION = ("Notations : e1, p1, e2, p2 (EN 1993-1-8 Tab. 3.3) ; e2,b, e1,b : pinces dans l'âme de la "
                    "poutre portée ; gA : trusquinage de l'aile A ; hc : hauteur des cornières ; zc : dessus de "
                    "la poutre portée → dessus des cornières ; c, dc : longueur et profondeur du grugeage ; "
                    "gh : jeu à l'about ; Δz : décalage des dessus de semelles ; z : excentricité ; p3 : entraxe "
                    "des files des deux cornières.")


def ec(txt):
    """Un texte du moteur en notation de l'Eurocode."""
    if not txt:
        return txt or ""
    for rx, rep in _REGLES:
        txt = rx.sub(rep, txt)
    return txt


# --- références courtes -----------------------------------------------
LEGENDE_REFERENCES = ("Références : EC3 = EN 1993-1-8 sauf indication (§, Tab.) ; EC3-1-1 = EN 1993-1-1 ; "
                      "MSB P5 ; P358 = SCI P358 ; ECCS = ECCS n°126.")

_REMPLACEMENTS = [
    ("EN 1993-1-8 ", ""), ("EN 1993-1-1 ", "EC3-1-1 "), ("MSB Part 5 ", "MSB P5 "),
    ("SCI P358 Check ", "P358 Ch. "), ("ECCS n°126", "ECCS"), ("Tableau ", "Tab. "),
    ("hypothèse de l'outil", "hyp. outil"),
]


def ref_courte(ref):
    """La référence courte d'une vérification (une ligne de tableau)."""
    parts = []
    for part in ref.split(" ; "):
        part = part.strip()
        if " : " in part:                       # « aires : MSB Part 5 §4.2.2.1 » → la source
            part = part.split(" : ", 1)[1]
        sources = re.findall(r"\((?:d'après )?((?:ECCS|SCI|MSB)[^)]*)\)", part)
        part = re.sub(r"\s*\([^)]*\)", "", part).strip()
        if part in ("modèle complémentaire", ""):
            continue
        for a, b in _REMPLACEMENTS:
            part = part.replace(a, b)
        parts.append(part)
        for s in sources:
            for a, b in _REMPLACEMENTS:
                s = s.replace(a, b)
            parts.append(s)
    vus = []
    for p in parts:
        if p not in vus:
            vus.append(p)
    return " · ".join(vus)


# --- alertes ----------------------------------------------------------
TITRES_ALERTES = {
    "S_e1bot": "Impossible — pince basse du groupe S",
    "P_e1bot": "Impossible — pince basse du groupe P",
    "S_e2": "Impossible — aile B trop courte (pince e2 du groupe S)",
    "P_e2": "Impossible — aile A trop courte (pince e2 du groupe P)",
    "zc_top": "Impossible — dessus des cornières dans le grugeage ou le congé",
    "h_dispo": "Impossible — hauteur disponible insuffisante",
    "dnt_min": "Grugeage supérieur insuffisant",
    "ln_min": "Longueur de grugeage insuffisante",
    "gap_neg": "Hors domaine — la poutre secondaire descend sous la principale",
    "dnb_min": "Grugeage inférieur nécessaire ou insuffisant",
    "P_web": "Impossible — boulons P hors de la partie droite de l'âme porteuse",
    "fillet": "Impossible — trou de boulon dans le congé de la cornière",
    "one_bolt": "Impossible — un seul boulon côté poutre secondaire",
    "d0_manq": "Trou surdimensionné : saisir d0",
    "MEd": "MEd ≠ 0 : traité comme moment parasite",
    "NEd": "NEd de traction : flexion hors plan de l'âme porteuse non vérifiée",
    "HEd": "HEd > 10 % de VEd : flexion hors plan des cornières non couverte",
    "soude2": "Soudée des deux côtés : capacité de rotation à justifier (§5.2)",
    "cat_cl": "Catégorie B ou C : boulons 8.8 ou 10.9 requis (§3.4.1)",
}


def titre_alerte(a):
    """Le titre court d'une alerte (Tableau 3.3 : la ligne en cause)."""
    if a.id.startswith("dist_"):
        return "Tab. 3.3 — " + a.msg.split(" : ", 1)[1].rstrip(".") if " : " in a.msg else a.msg
    return TITRES_ALERTES.get(a.id, a.msg)


def _premiere_phrase(txt):
    m = re.match(r"(.+?\.)(?:\s|$)", txt)
    return (m.group(1) if m else txt).rstrip(".")


def ligne_alerte(a, R):
    """``(titre, détail)`` : le détail est la première phrase chiffrée de
    l'explication, ou la ligne du Tableau 3.3 (valeur, limite)."""
    if a.id.startswith("dist_"):
        for x in R.dist:
            if x.alert == a.id:
                if x.val < x.min - 0.001:
                    det = F(x.val, 1) + " mm < " + F(x.kmin, 1) + "·d0 = " + F(x.min, 1) + " mm"
                else:
                    det = F(x.val, 1) + " mm > " + F(x.max, 0) + " mm"
                return titre_alerte(a), det
    return titre_alerte(a), ec(_premiere_phrase(a.why)) if a.why else ""

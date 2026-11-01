# -*- coding: utf-8 -*-
"""Le kit des notes PDF des assemblages : le bloc HYPOTHÈSES empaqueté.

Règle (charte du 31/10/2026, §11) : au plus ``n_max`` lignes de ``larg``
caractères, composées de SEGMENTS écrits par le module — jamais coupés ;
quand ils ne tiennent pas, les formes courtes sont prises du dernier
segment au premier ; s'ils ne tiennent toujours pas, ``ValueError`` (le
test rougit : c'est le texte qu'il faut revoir, pas la règle).
"""

import itertools

from acier.formats import F

_MINUSCULES = "abcdefghijklmnopqrstuvwxyzàâäéèêëîïôöùûüç"

# Les renvois aux guides : un § SANS indication est de l'EN 1993-1-8 (légende du pied de la note :
# « EC3 = EN 1993-1-8 sauf indication ») ; un § du guide MSB Part 5 porte donc TOUJOURS « MSB P5 »
# (relecture du 01/11/2026 : « (§4.2.4) » nu, à côté de « (§4.5.1) » de l'EN, se lisait comme l'EN).
MSB = "MSB P5 "


def coefficients(tete, g_M2, g_M2n, tete_net=None):
    """Les formes du segment des coefficients partiels (trois modules
    poutre-poutre), de la plus longue à la plus courte : ``tete``
    (« γM0 1,00 · γM2 1,25 ») suivie de ce que couvre γM2 ; quand γM2,net
    diffère, il est écrit à part, et ``tete_net`` remplace ``tete`` dans la
    forme courte (platine d'about : sans « (ANB) »)."""
    if abs(g_M2 - g_M2n) > 1e-9:
        net = " · γM2,net " + F(g_M2n, 2)
        return (tete + " (boulons, soudures)" + net + " (nettes)", (tete_net or tete) + net)
    return (tete + " (boulons, soudures, sections nettes)", tete + " (boulons, soudures, nettes)", tete)


def _capitale(s):
    """Capitale initiale d'une ligne — lettre latine seulement (« γM0 »
    reste tel quel)."""
    return s[0].upper() + s[1:] if s and s[0] in _MINUSCULES else s


def _emballer(textes, larg):
    """Glouton : un segment rejoint la ligne courante (« ; ») s'il tient
    avec le point final, sinon il ouvre la suivante. None si un segment seul
    ne tient pas."""
    lignes, cur = [], ""
    for t in textes:
        if len(t) + 1 > larg:
            return None
        cand = t if not cur else cur + " ; " + t
        if len(cand) + 1 <= larg:
            cur = cand
        else:
            lignes.append(_capitale(cur) + ".")
            cur = t
    if cur:
        lignes.append(_capitale(cur) + ".")
    return lignes


def empaqueter(segments, n_max=3, larg=110, une_fois=None):
    """``segments`` : chaînes, ou tuples de formes de la plus longue à la plus
    courte. Renvoie les lignes (≤ ``n_max``, chacune ≤ ``larg``).

    Le choix des formes : le PREMIER jeu, dans l'ordre lexicographique des
    indices de forme, dont l'emballage tient — les segments du début gardent
    leur forme longue tant que c'est possible, les formes courtes sont prises
    du dernier au premier. (Avant le 01/11/2026, un compteur sans remise à
    zéro : un segment raccourci laissait les suivants à leur forme la plus
    courte, et quatre cas figés ne trouvaient plus de solution dès que les
    renvois disaient « MSB P5 ».)

    ``une_fois`` (ex. ``MSB`` = « MSB P5 ») : un jeu où ce texte figure au
    plus une fois par ligne passe avant les autres ; à défaut, le premier
    jeu qui tient (le renvoi n'est jamais retiré, il peut alors figurer deux
    fois sur une ligne)."""
    formes = []
    for s in segments:
        fs = [s] if isinstance(s, str) else [f for f in s if f]
        if fs:
            formes.append(fs)
    bornes = []
    for fs in formes:                       # un segment qui ne tient pas seul prend d'abord sa forme qui tient
        i = 0
        while i < len(fs) and len(fs[i]) + 1 > larg:
            i += 1
        if i == len(fs):
            raise ValueError("segment trop long pour " + str(larg) + " caractères : " + fs[-1][:40])
        bornes.append(i)
    premier = None
    for choix in itertools.product(*[range(b, len(fs)) for b, fs in zip(bornes, formes)]):
        lignes = _emballer([fs[i] for fs, i in zip(formes, choix)], larg)
        if lignes is None or len(lignes) > n_max:
            continue
        if not une_fois or all(l.count(une_fois.strip()) <= 1 for l in lignes):
            return lignes
        premier = premier or lignes
    if premier is not None:
        return premier
    raise ValueError("les hypothèses ne tiennent pas en " + str(n_max) + " lignes de " + str(larg))

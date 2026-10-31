# -*- coding: utf-8 -*-
"""Le kit des notes PDF des assemblages : le bloc HYPOTHÈSES empaqueté.

Règle (charte du 31/10/2026, §11) : au plus ``n_max`` lignes de ``larg``
caractères, composées de SEGMENTS écrits par le module — jamais coupés ;
quand ils ne tiennent pas, les formes courtes sont prises du dernier
segment au premier ; s'ils ne tiennent toujours pas, ``ValueError`` (le
test rougit : c'est le texte qu'il faut revoir, pas la règle).
"""

_MINUSCULES = "abcdefghijklmnopqrstuvwxyzàâäéèêëîïôöùûüç"


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


def empaqueter(segments, n_max=3, larg=110):
    """``segments`` : chaînes, ou tuples de formes de la plus longue à la plus
    courte. Renvoie les lignes (≤ ``n_max``, chacune ≤ ``larg``)."""
    formes = []
    for s in segments:
        fs = [s] if isinstance(s, str) else [f for f in s if f]
        if fs:
            formes.append(fs)
    choix = []
    for fs in formes:                       # un segment qui ne tient pas seul prend d'abord sa forme qui tient
        i = 0
        while i < len(fs) and len(fs[i]) + 1 > larg:
            i += 1
        if i == len(fs):
            raise ValueError("segment trop long pour " + str(larg) + " caractères : " + fs[-1][:40])
        choix.append(i)
    while True:
        lignes = _emballer([fs[i] for fs, i in zip(formes, choix)], larg)
        if lignes is not None and len(lignes) <= n_max:
            return lignes
        for i in range(len(formes) - 1, -1, -1):
            if choix[i] + 1 < len(formes[i]):
                choix[i] += 1
                break
        else:
            raise ValueError("les hypothèses ne tiennent pas en " + str(n_max) + " lignes de " + str(larg))

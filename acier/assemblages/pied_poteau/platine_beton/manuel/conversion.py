# -*- coding: utf-8 -*-
"""Passage entre les deux modes de l'écran — calcul MANUEL (boulons et
éléments libres) et vérification COMPLÈTE (grille régulière centrée sur un
profilé) — sans perdre la disposition :

- ``vers_manuel(u)`` : la grille et le profilé deviennent une liste de
  boulons et un élément à l'axe (toujours possible) ;
- ``disposition_reguliere(u)`` : la disposition libre est-elle une grille
  centrée autour d'un seul profilé en I ou RHS à l'axe, sans raidisseur ?
  Si oui, ``vers_complet(u)`` la reporte dans les champs de la grille.
"""
from acier.bibliotheques import PERSO
from ..entrees import (MODE_MANUEL, MODE_COMPLET, FORME_I, FORME_RHS, TYPE_I, TYPE_RHS, element_defaut, defaults)
from .. import geometrie as GE
from . import elements as EL

TOL = 0.5          # mm : tolérance de position pour reconnaître une grille


def _complet(u):
    """Les saisies complétées par les défauts (une clé absente vaut son défaut)."""
    v = defaults()
    v.update({k: x for k, x in (u or {}).items()})
    return v


def vers_manuel(u):
    """Copie de ``u`` en mode manuel : boulons de la grille, profilé à l'axe."""
    u = _complet(u)
    v = dict(u)
    pos = GE.positions_tiges(u.get("n_ax") or 2, u.get("n_ay") or 2, u.get("w_x") or 0, u.get("w_y") or 0)
    v["tiges"] = [dict(x=float(x), y=float(y)) for (x, y, _i, _j) in pos]
    if u.get("forme") == FORME_RHS:
        e = element_defaut(TYPE_RHS)
        e.update(h=u.get("h_u") or 0, b=u.get("b_u") or 0, t=u.get("t_u") or 0)
    else:
        e = element_defaut(TYPE_I)
        e.update(prof=u.get("prof_c") or PERSO)
        if e["prof"] == PERSO:
            e.update(h=u.get("h_u") or 0, b=u.get("b_u") or 0, tw=u.get("tw_u") or 0, tf=u.get("tf_u") or 0,
                     r=u.get("r_u") or 0)
    v["elems"] = [e]
    v["raid"] = []
    v["mode"] = MODE_MANUEL
    return v


def _grille(vals):
    """``(n, w)`` si les valeurs distinctes forment une grille centrée régulière."""
    xs = sorted(set(round(float(x), 3) for x in vals))
    n = len(xs)
    if n == 1:
        return (1, 0.0) if abs(xs[0]) <= TOL else None
    w = xs[-1] - xs[0]
    if abs(xs[0] + xs[-1]) > TOL:
        return None
    for j, x in enumerate(xs):
        if abs(x - (-w / 2 + j * w / (n - 1))) > TOL:
            return None
    return n, w


def disposition_reguliere(u):
    """``dict(ok, raison, params)`` — ``params`` : les champs de la grille et
    du profilé pour la vérification complète."""
    u = _complet(u)
    elems = [EL.normaliser(e) for e in (u.get("elems") or [])]
    raid = u.get("raid") or []
    tiges = [(float(t.get("x") or 0), float(t.get("y") or 0)) for t in (u.get("tiges") or [])]
    if len(elems) != 1:
        return dict(ok=False, raison="un seul élément soudé est admis", params=None)
    e = elems[0]
    if e["type"] not in (TYPE_I, TYPE_RHS):
        return dict(ok=False, raison="l'élément doit être un profilé en I ou un profil creux rectangulaire", params=None)
    if abs(e["x"]) > TOL or abs(e["y"]) > TOL or e["rot"] != 0:
        return dict(ok=False, raison="le profilé doit être à l'axe de la platine, non tourné", params=None)
    if raid:
        return dict(ok=False, raison="pas de raidisseur dans la vérification complète", params=None)
    if not tiges:
        return dict(ok=False, raison="aucun boulon", params=None)
    gx = _grille([x for x, _y in tiges])
    gy = _grille([y for _x, y in tiges])
    if gx is None or gy is None or gx[0] * gy[0] != len(tiges):
        return dict(ok=False, raison="les boulons ne forment pas une grille centrée (n files × n rangées)", params=None)
    xs = sorted(set(round(x, 3) for x, _y in tiges))
    ys = sorted(set(round(y, 3) for _x, y in tiges))
    if len({(round(x, 3), round(y, 3)) for x, y in tiges}) != len(tiges) or \
            any((round(x, 3), round(y, 3)) not in {(a, b) for a in xs for b in ys} for x, y in tiges):
        return dict(ok=False, raison="les boulons ne forment pas une grille centrée (n files × n rangées)", params=None)
    if gx[0] < 1 or gy[0] < 1 or (gx[0] == 1 and gy[0] == 1):
        return dict(ok=False, raison="au moins deux boulons", params=None)
    p = dict(n_ax=gx[0], n_ay=gy[0], w_x=gx[1], w_y=gy[1])
    if e["type"] == TYPE_RHS:
        p.update(forme=FORME_RHS, h_u=e["h"], b_u=e["b"], t_u=e["t"])
    else:
        p.update(forme=FORME_I, prof_c=e["prof"])
        if e["prof"] == PERSO:
            p.update(h_u=e["h"], b_u=e["b"], tw_u=e["tw"], tf_u=e["tf"], r_u=e["r"])
    return dict(ok=True, raison="", params=p)


def vers_complet(u):
    """Copie de ``u`` en mode complet si la disposition est régulière, sinon None."""
    d = disposition_reguliere(u)
    if not d["ok"]:
        return None
    v = _complet(u)
    v.update(d["params"])
    v["mode"] = MODE_COMPLET
    return v

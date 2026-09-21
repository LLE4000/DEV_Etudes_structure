# -*- coding: utf-8 -*-
"""Instantané d'un résultat du moteur, dans la forme du corrigé de référence.

``instantane(R)`` reproduit la fonction ``snap`` de l'oracle Node
(``outils/oracle_double_corniere.js``) : mêmes clés, mêmes conventions
(valeurs non finies écrites « Infinity » / « -Infinity » / « NaN »). C'est
l'objet comparé par les tests de parité — et un format d'échange commode
(JSON) pour rejouer un calcul hors interface.
"""
import math


def num(v):
    """Nombre non fini → chaîne, comme dans le JSON de référence."""
    if isinstance(v, float) and not math.isfinite(v):
        if math.isnan(v):
            return "NaN"
        return "Infinity" if v > 0 else "-Infinity"
    return v


def _scalaire(v):
    return isinstance(v, (bool, int, float, str))


def _ligne_predim(r):
    prop = None
    if r.prop:
        prop = dict(reqs=[list(x) for x in r.prop.reqs], treq=num(r.prop.treq), tC=r.prop.tC,
                    gA=r.prop.gA, legreq=num(r.prop.legreq), leg=r.prop.leg, rC=r.prop.rC)
    return dict(boulon=r.b["n"], n=r.n, e1=r.e1, p1=r.p1, e2=r.e2, Lc=r.Lc, geom=r.geom,
                eta=num(r.eta), etas={k: num(v) for k, v in r.e.items()}, ok=r.ok,
                score=r.score, prop=prop)


def instantane(R, lignes_predim=True):
    """Le résultat ``R`` sous la forme ``attendu`` du corrigé de référence."""
    sc = {k: num(v) for k, v in R.items() if _scalaire(v)}
    p = R.pd
    pd = dict(boulon=p.boulon, n=p.n, e1=p.e1, p1=p.p1, e2=p.e2, LC=p.LC, eta=num(p.eta),
              treq=num(p.treq), tC=p.tC, gA=p.gA, legreq=num(p.legreq), leg=p.leg, rC=p.rC,
              found=p.found, hav=num(p.hav), msg=p.msg,
              reqs=[[x[0], num(x[1])] for x in p.reqs])
    if lignes_predim:
        pd["lignes"] = [_ligne_predim(r) for r in p.rows]
    return dict(
        scalaires=sc, statut=R.statut, verified=R.verified, eta_max=num(R.eta_max),
        gouvernante=R.gov.key if R.gov else None,
        verifications=[dict(key=c.key, active=c.active, Ed=num(c.Ed), Rd=num(c.Rd),
                            eta=num(c.eta), ok=c.ok, vals=c.vals) for c in R.checks],
        alertes=[dict(id=a.id, bloquant=a.block, message=a.msg, explication=a.why,
                      champs=list(a.fields), cotes=list(a.dims), elements=list(a.elems))
                 for a in R.alerts],
        tableau_3_3=[dict(libelle=x.lab, valeur=num(x.val), min=num(x.min),
                          max=num(x.max) if x.max is not None else None, ok=x.ok)
                     for x in R.dist],
        efforts_boulons_S=[dict(i=b.i, j=b.j, x=num(b.x), y=num(b.y), fx=num(b.fx),
                                fz=num(b.fz), f=num(b.f)) for b in R.tabS],
        efforts_boulons_P=[dict(i=b.i, j=b.j, x=num(b.x), y=num(b.y), fx=num(b.fx),
                                fz=num(b.fz), f=num(b.f)) for b in R.tabP],
        predim=pd)


def comparer(attendu, obtenu, tol=1e-9, chemin=""):
    """Compare deux instantanés : écart relatif ≤ ``tol`` sur les nombres,
    égalité stricte sur les booléens, chaînes, None. Retourne
    ``(nombre_de_grandeurs_comparées, [écarts])`` ; une clé ``lignes`` absente
    du corrigé est ignorée (le JSON ne la fournit que pour certains cas)."""
    n = 0
    ecarts = []

    def rec(a, b, path):
        nonlocal n
        if isinstance(a, dict) and isinstance(b, dict):
            for k in a:
                if k not in b:
                    ecarts.append(f"{path}.{k} : absent du résultat")
                    continue
                rec(a[k], b[k], f"{path}.{k}")
            for k in b:
                if k not in a and not (k == "lignes"):
                    ecarts.append(f"{path}.{k} : en trop dans le résultat")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                ecarts.append(f"{path} : {len(a)} éléments attendus, {len(b)} obtenus")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                rec(x, y, f"{path}[{i}]")
        else:
            n += 1
            if a is None and b in ("Infinity", "-Infinity", "NaN"):
                # l'oracle ne convertit en chaîne que les grandeurs de premier
                # niveau ; un nombre non fini IMBRIQUÉ (etas, reqs, efforts par
                # boulon) sort de JSON.stringify comme null : même valeur
                return
            if isinstance(a, bool) or isinstance(b, bool):
                if a is not b:
                    ecarts.append(f"{path} : {a!r} attendu, {b!r} obtenu")
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if a != b:
                    d = abs(a - b) / max(abs(a), abs(b), 1e-300)
                    if d > tol:
                        ecarts.append(f"{path} : {a!r} attendu, {b!r} obtenu (écart relatif {d:.2e})")
            elif a != b:
                ecarts.append(f"{path} : {a!r} attendu, {b!r} obtenu")

    rec(attendu, obtenu, chemin)
    return n, ecarts

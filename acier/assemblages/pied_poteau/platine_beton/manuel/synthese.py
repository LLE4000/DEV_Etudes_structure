# -*- coding: utf-8 -*-
"""Synthèse des vérifications du calcul manuel, regroupées par élément
physique — même structure que la vérification complète (écran + note).
Rien n'est recalculé. Les lignes d'information (tiges acier, appui) sont
affichées, marquées, et ne comptent ni dans le taux d'un élément ni dans le
verdict : le verdict porte sur la platine."""
from acier.formats import F, pct
from ..notation import ref_courte
from .moteur import INFO_SEUL

TABLES = [
    ("platine", "Platine — épaisseur requise et ses quatre critères",
     [("tpq", "Épaisseur requise tp"),
      ("ptl", "Tendue — tronçon en T du boulon le plus sollicité"),
      ("pts", "Tendue — section au nu de la paroi"),
      ("pc", "Comprimée — bande au nu de la face")]),
    ("trous", "Trous de la platine",
     [("pb", "Pression diamétrale (Tab. 3.4)")]),
    ("tiges", "Tiges d'ancrage (acier) — information",
     [("ta", "Traction"), ("tv", "Cisaillement"), ("tvn", "Interaction traction / cisaillement")]),
    ("appui", "Appui sur le béton — information",
     [("bet", "Pression maximale ≤ fjd"), ("ncp", "Compression N ≤ fjd·Aeff")]),
]
ORDRE = tuple(t[0] for t in TABLES)
COURT = {k: l for _, _, ls in TABLES for k, l in ls}

ELEMENTS = [("Platine", "Platine"), ("Trous", "Trous"), ("Tiges", "Tiges"), ("Appui", "Appui")]


def court(c):
    return COURT.get(c.key, c.label)


def info(R, k):
    return k in getattr(R, "info_keys", INFO_SEUL)


def taux_par_element(R):
    """``(nom, η, ok, pire)`` par élément ; les lignes d'information ne fixent
    pas de taux."""
    out = []
    for grp, nom in ELEMENTS:
        on = [c for c in R.checks if c.grp == grp and c.active and not info(R, c.key)]
        if on:
            pire = max(on, key=lambda c: c.eta)
            out.append((nom, pire.eta, all(c.ok for c in on), pire))
    return out


def pire_information(R):
    """La ligne d'information la plus sollicitée (tiges, appui) — pour le
    dire à l'écran et sur la note, hors verdict."""
    hors = [c for c in R.checks if c.active and info(R, c.key)]
    return max(hors, key=lambda c: c.eta) if hors else None


def ed_rd(c):
    nd = 3 if c.unit == "-" else 1
    return F(c.Ed, nd), F(c.Rd, nd), ("" if c.unit == "-" else c.unit)


def lignes(R, cle):
    out = []
    for _, _, ls in [t for t in TABLES if t[0] == cle]:
        for k, lab in ls:
            c = R.ck.get(k)
            if c is None or not c.active:
                continue
            ed, rd, un = ed_rd(c)
            out.append(dict(key=k, lab=lab, Ed=ed, Rd=rd, unit=un, eta=c.eta, pct=pct(c.eta, 1),
                            ok=bool(c.ok), ref=ref_courte(c.ref), nat=c.nat, c=c, info=info(R, k)))
    return out


def titre_table(cle, R=None):
    return {t[0]: t[1] for t in TABLES}[cle]

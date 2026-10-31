# -*- coding: utf-8 -*-
"""Synthèse des vérifications du pied de poteau, regroupées par élément
physique — même structure que les assemblages poutre–poutre (écran + note).
Rien n'est recalculé."""
from acier.formats import F, pct
from .notation import ref_courte
from .moteur import DETAIL_SEUL, INFO_SEUL

TABLES = [
    ("pied", "Pied — flexion composée N + Mx + My",
     [("fc1", "Niveau 1 : interaction linéaire (Tab. 6.7)"),
      ("fc2", "Niveau 2 : domaine exact (information)")]),
    ("beton", "Béton — appui et ancrage",
     [("ncp", "Compression : N ≤ fjd·Aeff"), ("bet", "Pression maximale ≤ fjd (information)"),
      ("cone", "Cône de béton du groupe tendu"), ("pull", "Arrachement (tête) / adhérence (crosse)"),
      ("edge", "Rupture du bord en cisaillement"), ("pry", "Pry-out du groupe"),
      ("cv", "Interaction N / V du béton")]),
    ("platine", "Platine",
     [("pc", "Comprimée — flexion du débord"), ("pt", "Tendue — tronçon en T")]),
    ("tiges", "Tiges d'ancrage (acier)",
     [("ta", "Traction"), ("tv", "Cisaillement"), ("tvn", "Interaction traction / cisaillement")]),
    ("poteau", "Poteau",
     [("cfc", "Aile / paroi comprimée")]),
    ("soudure", "Cordons poteau / platine",
     [("w", "Contrainte équivalente"), ("wp", "Contrainte normale σ⊥"), ("wa", "Gorge minimale")]),
]
COURT = {k: l for _, _, ls in TABLES for k, l in ls}

ELEMENTS = [("Pied", "Pied"), ("Béton", "Béton"), ("Platine", "Platine"), ("Tiges", "Tiges"),
            ("Poteau", "Poteau"), ("Soudure", "Soudure")]


def court(c):
    return COURT.get(c.key, c.label)


def taux_par_element(R):
    """``(nom, η, ok, pire)`` par élément ; une ligne d'information (niveau 2,
    pression maximale) ou un minimum de détail satisfait ne fixe pas le taux
    de son élément."""
    out = []
    for grp, nom in ELEMENTS:
        on = [c for c in R.checks if c.grp == grp and c.active]
        if on:
            utiles = [c for c in on if c.key not in INFO_SEUL and not (c.key in DETAIL_SEUL and c.ok)] or on
            pire = max(utiles, key=lambda c: c.eta)
            out.append((nom, pire.eta, all(c.ok for c in on if c.key not in INFO_SEUL), pire))
    return out


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
                            ok=bool(c.ok), ref=ref_courte(c.ref), nat=c.nat, c=c))
    return out


def titre_table(cle):
    return {t[0]: t[1] for t in TABLES}[cle]

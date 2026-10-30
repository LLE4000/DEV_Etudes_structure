# -*- coding: utf-8 -*-
"""Synthèse des vérifications de la platine d'about, regroupées par élément
physique — même structure que les deux autres assemblages (écran + note).
Rien n'est recalculé."""
from acier.formats import F, pct
from .notation import ref_courte
from .moteur import DETAIL_SEUL

TABLES = [
    ("boulons", "Boulons",
     [("bg", "Groupe : cisaillement et pression diam."), ("gl", "Glissement"),
      ("bvt", "Cisaillement + traction")]),
    ("platine", "Platine d'about",
     [("pcg", "Cisaillement brut"), ("pcn", "Cisaillement net"), ("pcb", "Rupture de bloc"),
      ("ptT", "Flexion en traction (tronçon en T)"), ("pp", "Poinçonnement")]),
    ("portee", "Poutre secondaire (portée)",
     [("vwS", "Âme – cisaillement sur hp"), ("vgS", "Grugeage – cisaillement"),
      ("mN", "Grugeage – flexion + V"), ("stN", "Grugeage – stabilité (c)"),
      ("stD", "Grugeage – stabilité (dc)"), ("tnS", "Âme – traction")]),
    ("porteuse", "Poutre principale (porteuse)",
     [("pdP", "Pression diamétrale – âme"), ("vlP", "Cisaillement local brut"),
      ("vnP", "Cisaillement local net")]),
    ("soudure", "Soudure (double cordon)",
     [("w", "Effort au point critique"), ("wT", "Tying"), ("wa", "Gorge minimale")]),
]
COURT = {k: l for _, _, ls in TABLES for k, l in ls}

ELEMENTS = [("Boulons", "Boulons"), ("Platine", "Platine"), ("Poutre secondaire", "Portée"),
            ("Poutre principale", "Porteuse"), ("Soudure", "Soudure")]


def court(c):
    return COURT.get(c.key, c.label)


def taux_par_element(R):
    """``(nom, η, ok, pire)`` par élément ; un minimum de détail satisfait
    (gorge ≥ 3 mm) ne fixe pas le taux de son élément."""
    out = []
    for grp, nom in ELEMENTS:
        on = [c for c in R.checks if c.grp == grp and c.active]
        if on:
            utiles = [c for c in on if not (c.key in DETAIL_SEUL and c.ok)] or on
            pire = max(utiles, key=lambda c: c.eta)
            out.append((nom, pire.eta, all(c.ok for c in on), pire))
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

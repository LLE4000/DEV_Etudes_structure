# -*- coding: utf-8 -*-
"""Synthèse des vérifications du plat d'âme, regroupées par élément physique
— même structure que le module doubles cornières (écran + note). Rien n'est
recalculé."""
from acier.formats import F, pct
from .notation import ref_courte

TABLES = [
    ("boulons", "Boulons", [("bv", "Cisaillement du groupe"), ("gl", "Glissement")]),
    ("plat", "Plat d'âme",
     [("pdL", "Pression diamétrale"), ("pcg", "Cisaillement brut"), ("pcn", "Cisaillement net"),
      ("pcb", "Rupture de bloc"), ("pfl", "Flexion dans le plan"), ("plt", "Déversement (plat long)"),
      ("ptn", "Traction nette")]),
    ("portee", "Poutre secondaire (portée)",
     [("pdS", "Pression diamétrale – âme"), ("vgS", "Cisaillement brut – âme"), ("vnS", "Cisaillement net – âme"),
      ("vbS", "Rupture de bloc – âme"), ("mN", "Grugeage – flexion + V"), ("m2", "Grugeage – 2e file"),
      ("stN", "Grugeage – stabilité (c)"), ("stD", "Grugeage – stabilité (dc)"), ("tnS", "Traction nette – âme")]),
    ("porteuse", "Poutre principale (porteuse)", [("vlP", "Cisaillement local de l'âme")]),
    ("soudure", "Soudure (double cordon)",
     [("w", "Effort au point critique"), ("wa", "Gorge minimale")]),
]
COURT = {k: l for _, _, ls in TABLES for k, l in ls}

ELEMENTS = [("Boulons", "Boulons"), ("Plat d'âme", "Plat"), ("Poutre secondaire", "Portée"),
            ("Poutre principale", "Porteuse"), ("Soudure", "Soudure")]


def court(c):
    return COURT.get(c.key, c.label)


def taux_par_element(R):
    out = []
    for grp, nom in ELEMENTS:
        on = [c for c in R.checks if c.grp == grp and c.active]
        if on:
            pire = max(on, key=lambda c: c.eta)
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


def refs_fusionnees(refs):
    vus = []
    for r in refs:
        for p in r.split(" · "):
            if p not in vus:
                vus.append(p)
    return " · ".join(vus)

# -*- coding: utf-8 -*-
"""Synthèse des vérifications du plat d'âme sur poteau, regroupées par
élément physique — même structure que les modules poutre–poutre (écran +
note). Rien n'est recalculé.

L'ordre des tables est celui de la liste ordonnée de l'étude
(`POUTRE_POTEAU.md` §8.1) : Boulons · Plat d'âme · Poutre portée · Poteau
porteur · Soudure. En cas (a) le groupe « Poteau porteur » ne porte AUCUNE
vérification active (§4.4) : ``taux_par_element`` ne le rend pas et
``lignes`` est vide — l'écran l'écrit dans « Périmètre de la vérification »."""
from acier.formats import F, pct
from .notation import ref_courte

TABLES = [
    ("boulons", "Boulons",
     [("bv", "Cisaillement du groupe"), ("gl", "Glissement"),
      ("bvT", "Tying – cisaillement")]),
    ("plat", "Plat d'âme",
     [("pdL", "Pression diamétrale"), ("pcg", "Cisaillement brut"), ("pcn", "Cisaillement net"),
      ("pcb", "Rupture de bloc"), ("pfl", "Flexion dans le plan"), ("plt", "Déversement (plat long)"),
      ("ptn", "Traction nette"), ("pbt", "Bloc en traction")]),
    ("portee", "Poutre portée",
     [("pdS", "Pression diamétrale – âme"), ("vgS", "Cisaillement brut – âme"), ("vnS", "Cisaillement net – âme"),
      ("vbS", "Rupture de bloc – âme"), ("mN", "Grugeage – flexion + V"), ("m2", "Grugeage – 2e file"),
      ("stN", "Grugeage – stabilité (c)"), ("stD", "Grugeage – stabilité (dc)"),
      ("tnS", "Traction nette – âme"), ("tbS", "Bloc en traction – âme")]),
    ("poteau", "Poteau porteur", [("vlP", "Cisaillement local de l'âme")]),
    ("soudure", "Soudure (double cordon)",
     [("w", "Effort au point critique"), ("wT", "Tying – cordons"), ("wa", "Gorge minimale")]),
]
COURT = {k: l for _, _, ls in TABLES for k, l in ls}

ELEMENTS = [("Boulons", "Boulons"), ("Plat d'âme", "Plat"), ("Poutre portée", "Portée"),
            ("Poteau porteur", "Poteau"), ("Soudure", "Soudure")]


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

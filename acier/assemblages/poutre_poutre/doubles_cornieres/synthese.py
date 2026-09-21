# -*- coding: utf-8 -*-
"""Synthèse des vérifications, regroupées par élément physique — la même
structure pour l'écran (onglet Vérifications) et la note d'une page.

Rien n'est recalculé : les lignes lisent ``Ed``, ``Rd``, ``eta``, ``ok`` de
chaque ``Verification`` ; le taux d'un élément est le plus grand des taux
que le moteur a nommés dans son groupe.

- ``TABLES``               : les cinq tableaux (clé, titre, lignes) ;
- ``MATRICE_CORNIERES``    : lignes de la matrice aile A · aile B · max ;
- ``taux_par_element(R)``  : ``[(élément, η, ok), …]`` ;
- ``lignes(R, cle)``       : les lignes actives d'un tableau ;
- ``matrice_cornieres(R)`` : les lignes actives de la matrice.
"""
from acier.formats import F, pct
from .notation import ref_courte, ec

# Libellés courts des vérifications, par tableau
TABLES = [
    ("boulons", "Boulons", [("bv_S", "Cisaillement – groupe S"), ("bv_P", "Cisaillement – groupe P"),
                            ("bt_P", "Traction – groupe P"), ("bi_P", "Interaction V + T – groupe P"),
                            ("gl_S", "Glissement – groupe S"), ("gl_P", "Glissement – groupe P")]),
    ("portee", "Poutre secondaire (portée)",
     [("pdS", "Pression diamétrale – âme"), ("vgS", "Cisaillement brut – âme"), ("vnS", "Cisaillement net – âme"),
      ("vbS", "Rupture de bloc – âme"), ("mN", "Grugeage – flexion + V"), ("m2", "Grugeage – 2e file"),
      ("stN", "Grugeage – stabilité (c)"), ("stD", "Grugeage – stabilité (dc)"),
      ("tnS", "Traction nette – âme"), ("tbS", "Bloc en traction – âme")]),
    ("porteuse", "Poutre principale (porteuse)",
     [("pdP", "Pression diamétrale – âme"), ("vlP", "Cisaillement local brut"), ("vnP", "Cisaillement local net")]),
    ("cordons", "Cordons d'angle",
     [("wS", "Ailes B – effort au point critique"), ("waS", "Ailes B – gorge minimale"), ("wlS", "Ailes B – longueur minimale"),
      ("wP", "Ailes A – effort au point critique"), ("waP", "Ailes A – gorge minimale"), ("wlP", "Ailes A – longueur minimale")]),
]
# Matrice des cornières : (libellé, clé aile A, clé aile B)
MATRICE_CORNIERES = [("Pression diamétrale", "pdA", "pdB"), ("Cisaillement brut", "cgA", "cgB"),
                     ("Cisaillement net", "cnA", "cnB"), ("Rupture de bloc", "cbA", "cbB"),
                     ("Flexion dans le plan", "flA", "flB"), ("Traction nette", None, "tnB"),
                     ("Bloc en traction", None, "tbB"), ("Tronçon en T (NEd)", "tsA", None)]
COURT = {k: l for _, _, ls in TABLES for k, l in ls}
for _l, _a, _b in MATRICE_CORNIERES:
    if _a:
        COURT[_a] = "Aile A – " + _l[0].lower() + _l[1:]
    if _b:
        COURT[_b] = "Aile B – " + _l[0].lower() + _l[1:]

ELEMENTS = [("Boulons", "Boulons"), ("Cornières", "Cornières"), ("Poutre secondaire", "Portée"),
            ("Poutre principale", "Porteuse"), ("Soudures", "Cordons")]


def court(c):
    return COURT.get(c.key, ec(c.label))


def taux_par_element(R):
    """``[(élément, η maximal, tout vérifié), …]`` pour les groupes qui ont au
    moins une vérification active."""
    out = []
    for grp, nom in ELEMENTS:
        on = [c for c in R.checks if c.grp == grp and c.active]
        if on:
            pire = max(on, key=lambda c: c.eta)
            out.append((nom, pire.eta, all(c.ok for c in on), pire))
    return out


def ed_rd(c):
    """« Ed / Rd unité » tels que le moteur les donne."""
    nd = 3 if c.unit == "-" else 1
    return F(c.Ed, nd), F(c.Rd, nd), ("" if c.unit == "-" else c.unit)


def lignes(R, cle):
    """Les lignes actives d'un tableau : ``dict(key, lab, Ed, Rd, unit, eta, ok, ref, nat, c)``."""
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


def matrice_cornieres(R):
    """Les lignes actives de la matrice : ``dict(lab, A, B, max)`` où A et B
    sont des lignes (voir ``lignes``) ou None, et max la plus sollicitée."""
    out = []
    for lab, ka, kb in MATRICE_CORNIERES:
        cells = {}
        for col, k in (("A", ka), ("B", kb)):
            c = R.ck.get(k) if k else None
            if c is not None and c.active:
                ed, rd, un = ed_rd(c)
                cells[col] = dict(key=k, Ed=ed, Rd=rd, unit=un, eta=c.eta, pct=pct(c.eta, 1), ok=bool(c.ok),
                                  ref=ref_courte(c.ref), nat=c.nat, c=c)
            else:
                cells[col] = None
        if not cells["A"] and not cells["B"]:
            continue
        pire = max([x for x in cells.values() if x], key=lambda x: x["eta"])
        out.append(dict(lab=lab, A=cells["A"], B=cells["B"], max=pire))
    return out


def titre_table(cle):
    return {t[0]: t[1] for t in TABLES}[cle]


def lignes_ref(c):
    """La référence courte d'une vérification."""
    return ref_courte(c.ref)


def refs_fusionnees(refs):
    """Plusieurs références courtes en une, sans doublon (« Tab. 3.4 · MSB §4.2.1.1 »)."""
    vus = []
    for r in refs:
        for p in r.split(" · "):
            if p not in vus:
                vus.append(p)
    return " · ".join(vus)

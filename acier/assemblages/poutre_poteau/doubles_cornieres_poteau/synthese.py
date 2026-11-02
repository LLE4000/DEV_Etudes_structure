# -*- coding: utf-8 -*-
"""Synthèse des vérifications des doubles cornières sur poteau, regroupées
par élément physique — même structure que les modules poutre–poutre (écran +
note). Rien n'est recalculé.

L'ordre des tables est celui de la liste ordonnée de l'étude
(`POUTRE_POTEAU.md` §8.2) : Boulons · Cornières · Poutre portée · Poteau
porteur · Soudures. La MATRICE des cornières (aile A / aile B / max) est
celle du grand frère : deux colonnes, une ligne par vérification.

Le groupe « Poteau porteur » n'est jamais vide — la pression diamétrale de
l'aile ou de l'âme y est toujours active quand les cornières sont boulonnées
sur le poteau ; en revanche le cisaillement local (``vlP``, ``vnP``) ne vit
qu'en cas (b), et le tronçon en T de l'aile (``tfP``) et l'âme en traction
transversale (``twtP``) qu'en cas (a) avec des boulons tendus."""
from acier.formats import F, pct
from .notation import ref_courte

TABLES = [
    ("boulons", "Boulons",
     [("bv_S", "Cisaillement – groupe S"), ("bv_P", "Cisaillement – groupe P"),
      ("bt_P", "Traction – groupe P"), ("bi_P", "Interaction V + T – groupe P"),
      ("gl_S", "Glissement – groupe S"), ("gl_P", "Glissement – groupe P"),
      ("bvT_S", "Tying – cisaillement du groupe S")]),
    # table PLATE des cornières, dans l'ordre du §8.2 : c'est elle que la NOTE
    # imprime (une ligne par vérification) ; l'écran rend la même chose en
    # MATRICE aile A / aile B (``matrice_cornieres``), plus compacte.
    ("cornieres", "Cornières",
     [("pdB", "Aile B – pression diamétrale"), ("pdA", "Aile A – pression diamétrale"),
      ("cgB", "Aile B – cisaillement brut"), ("cnB", "Aile B – cisaillement net"),
      ("cbB", "Aile B – rupture de bloc"), ("flB", "Aile B – flexion dans le plan"),
      ("cgA", "Aile A – cisaillement brut"), ("cnA", "Aile A – cisaillement net"),
      ("cbA", "Aile A – rupture de bloc"), ("flA", "Aile A – flexion dans le plan"),
      ("tsA", "Aile A – tronçon en T (NEd)"), ("tnB", "Aile B – traction nette"),
      ("tbB", "Aile B – bloc en traction")]),
    ("portee", "Poutre portée",
     [("pdS", "Pression diamétrale – âme"), ("vgS", "Cisaillement brut – âme"),
      ("vnS", "Cisaillement net – âme"), ("vbS", "Rupture de bloc – âme"),
      ("mN", "Grugeage – flexion + V"), ("m2", "Grugeage – 2e file"),
      ("stN", "Grugeage – stabilité (c)"), ("stD", "Grugeage – stabilité (dc)"),
      ("tnS", "Traction nette – âme"), ("tbS", "Bloc en traction – âme")]),
    ("poteau", "Poteau porteur",
     [("pdP", "Pression diamétrale – aile ou âme"), ("vlP", "Cisaillement local brut – âme"),
      ("vnP", "Cisaillement local net – âme"), ("tfP", "Aile en flexion (tronçon en T)"),
      ("twtP", "Âme en traction transversale"), ("ppP", "Poinçonnement")]),
    ("cordons", "Cordons d'angle",
     [("wS", "Ailes B – effort au point critique"), ("waS", "Ailes B – gorge minimale"),
      ("wlS", "Ailes B – longueur minimale"), ("wtS", "Ailes B – tying"),
      ("wP", "Ailes A – effort au point critique"), ("waP", "Ailes A – gorge minimale"),
      ("wlP", "Ailes A – longueur minimale"), ("wtP", "Ailes A – tying")]),
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

ELEMENTS = [("Boulons", "Boulons"), ("Cornières", "Cornières"), ("Poutre portée", "Portée"),
            ("Poteau porteur", "Poteau"), ("Soudures", "Cordons")]


def court(c):
    return COURT.get(c.key, c.label)


def taux_par_element(R):
    """``[(élément, η maximal, tout vérifié, pire), …]`` pour les groupes qui
    ont au moins une vérification active."""
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


def matrice_cornieres(R):
    """Les lignes actives de la matrice : ``dict(lab, A, B, max)`` où A et B
    sont des cellules (ou None), et max la plus sollicitée."""
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


def refs_fusionnees(refs):
    vus = []
    for r in refs:
        for p in r.split(" · "):
            if p not in vus:
                vus.append(p)
    return " · ".join(vus)

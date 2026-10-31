# -*- coding: utf-8 -*-
"""Synthèse des vérifications du pied de poteau, regroupées par élément
physique — même structure que les assemblages poutre–poutre (écran + note).
Rien n'est recalculé. Les vérifications hors périmètre (béton d'ancrage en
mode EC3) ou d'information (niveau 2, pression maximale) sont affichées,
marquées, et ne comptent ni dans le taux d'un élément ni dans le verdict."""
from acier.formats import F, pct
from .notation import ref_courte
from .moteur import DETAIL_SEUL, INFO_SEUL

TABLES = [
    ("pied", "Pied — flexion composée N + Mx + My",
     [("fc1", "Niveau 1 : interaction linéaire (Tab. 6.7)"),
      ("fc2", "Niveau 2 : domaine exact (information)")]),
    ("platine", "Platine",
     [("tpq", "Épaisseur requise tp"),
      ("pc", "Comprimée — flexion du débord"), ("pt", "Tendue — tronçon en T")]),
    ("beton", "Béton d'appui (EC3 §6.2.5)",
     [("ncp", "Compression : N ≤ fjd·Aeff"), ("bet", "Pression maximale ≤ fjd (information)")]),
    ("ancrages", "Ancrages dans le béton (EN 1992-4)",
     [("cone", "Cône de béton du groupe tendu"), ("pull", "Arrachement (tête) / adhérence (crosse, scellée)"),
      ("edge", "Rupture du bord en cisaillement"), ("pry", "Pry-out du groupe"),
      ("cv", "Interaction N / V du béton")]),
    ("tiges", "Tiges d'ancrage (acier)",
     [("ta", "Traction"), ("tv", "Cisaillement"), ("tvn", "Interaction traction / cisaillement")]),
    ("poteau", "Poteau",
     [("cfc", "Aile / paroi comprimée")]),
    ("soudure", "Cordons poteau / platine",
     [("w", "Contrainte équivalente"), ("wp", "Contrainte normale σ⊥"), ("wa", "Gorge minimale")]),
]
ORDRE = tuple(t[0] for t in TABLES)
COURT = {k: l for _, _, ls in TABLES for k, l in ls}

ELEMENTS = [("Pied", "Pied"), ("Platine", "Platine"), ("Béton", "Appui"), ("Ancrages", "Ancrages"),
            ("Tiges", "Tiges"), ("Poteau", "Poteau"), ("Soudure", "Soudure")]


def court(c):
    return COURT.get(c.key, c.label)


def info(R, k):
    """Vérification affichée en information : hors périmètre ou renseignement."""
    return k in getattr(R, "info_keys", INFO_SEUL)


def taux_par_element(R):
    """``(nom, η, ok, pire)`` par élément ; une ligne d'information (niveau 2,
    pression maximale, béton d'ancrage hors périmètre) ou un minimum de
    détail satisfait ne fixe pas le taux de son élément ; un élément qui n'a
    que des lignes d'information n'apparaît pas."""
    out = []
    for grp, nom in ELEMENTS:
        on = [c for c in R.checks if c.grp == grp and c.active and not info(R, c.key)]
        if on:
            utiles = [c for c in on if not (c.key in DETAIL_SEUL and c.ok)] or on
            pire = max(utiles, key=lambda c: c.eta)
            out.append((nom, pire.eta, all(c.ok for c in on), pire))
    return out


def pire_information(R):
    """La vérification d'information la plus sollicitée du béton d'ancrage
    (mode EC3) — pour le dire à l'écran et sur la note, hors verdict."""
    hors = [c for c in R.checks if c.grp == "Ancrages" and c.active and info(R, c.key)]
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
            if info(R, k) and "(information)" not in lab and cle != "ancrages":
                lab = lab + " (information)"
            out.append(dict(key=k, lab=lab, Ed=ed, Rd=rd, unit=un, eta=c.eta, pct=pct(c.eta, 1),
                            ok=bool(c.ok), ref=ref_courte(c.ref), nat=c.nat, c=c, info=info(R, k)))
    return out


def titre_table(cle, R=None):
    """Le titre d'un tableau ; celui des ancrages dit « information, hors
    verdict » quand le périmètre est l'EN 1993-1-8 seule."""
    t = {t[0]: t[1] for t in TABLES}[cle]
    if cle == "ancrages" and R is not None and getattr(R, "perim_ec3", False):
        t += " — information, hors verdict"
    return t

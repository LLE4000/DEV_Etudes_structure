# -*- coding: utf-8 -*-
"""NON-RÉGRESSION DU CALCUL après la refonte de l'écran et de la note.

Le moteur ne devait pas changer : pour chacun des 31 cas de référence
(``acier/reference/reference_double_corniere.json``, figé avant la refonte,
commit 52b5e4e), on compare au corrigé chaque grandeur nommée par le moteur —
scalaires, Ed / Rd / η / statut de chaque vérification, vérification
dimensionnante, alertes (message, explication, champs, cotes), Tableau 3.3,
efforts par boulon, prédimensionnement — puis les cinq exports texte.

Le test imprime « RÉGRESSION CALCUL : 0 différence » ou liste chaque
différence. Il rougit si on retire une grandeur ou si une valeur bouge.

Lot A (16/11/2026) : le corrigé garde SON matériau (EN 1993-1-1 tableau 3.1,
sans épaisseur), que le test lui donne (``parite.materiau_du_corrige``) ;
les notes du site prennent la table belge (tableau 3.3 ANB) — écart nommé,
prouvé par tests/test_assemblages_parite.py §4. Trois NOTES du site sont
figées sur cette table (tests/ref_doubles_cornieres.json : note neuve, IPE 600
sur HEB 600, S275 grugée), avec des lignes recalculées à la main et la
mutation « ancienne table » qui doit les faire rougir.

Lancement : python3 tests/test_assemblages_regression.py
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages.poutre_poutre.doubles_cornieres import moteur, parite, texte  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


with open("acier/reference/reference_double_corniere.json", encoding="utf-8") as fh:
    REF = json.load(fh)
CORRIGE = parite.materiau_du_corrige(REF)

print("=== Résultats du moteur : 31 cas de référence ===")
n_tot = 0
diffs = []
for cas in REF["cas"]:
    R = moteur.compute(cas["inputs"], CORRIGE)
    n, ecarts = parite.comparer(cas["attendu"], parite.instantane(R), chemin=cas["id"])
    n_tot += n
    diffs.extend(ecarts)
    # le corrigé garde les mêmes vérifications, dans le même ordre
    if [v["key"] for v in cas["attendu"]["verifications"]] != [c.key for c in R.checks]:
        diffs.append(cas["id"] + " : liste des vérifications différente")
chk(f"{len(REF['cas'])} cas, {n_tot} grandeurs comparées", n_tot > 20000 and not diffs, " || ".join(diffs[:5]))

print("\n=== Exports texte ===")
for ident, versions in REF["exports_texte"].items():
    inputs = {c["id"]: c["inputs"] for c in REF["cas"]}[ident] if ident != "defaut" else {}
    R = moteur.compute(inputs, CORRIGE)
    for nom, complet in (("essentiel", False), ("complet", True)):
        if nom in versions:
            t = texte.construire_texte(R, complet)
            if t.strip() != versions[nom].strip():
                diffs.append(f"export {ident} {nom} : texte différent")
chk("les exports texte du corrigé sont reproduits", not any(d.startswith("export") for d in diffs))

print("\n=== Notes du SITE (table belge) : trois cas figés (tests/ref_doubles_cornieres.json) ===")
# Le corrigé tourne sur SON matériau : sans ces cas, aucune valeur d'une NOTE des doubles cornières poutre–poutre
# n'était figée sur la table du site (NBN EN 1993-1-1 ANB:2018 3.2.1(1), option a ; tableau 3.3 ANB). Trois notes :
# la note neuve (porteuse HEA 400, tf = 19 mm → f_y = 345 ; portée HEA 300), une IPE 600 portée par une HEB 600
# (tf = 19 et 30 mm → 345), une S275 grugée (f_y = 265 de 16 à 40 mm, f_u = 410 ; domaine lu sur la nuance : 54,3).
# Gelées par le moteur le 16/11/2026 (mêmes clés que tests/ref_plat_ame.json) ; une ligne de chaque cas est
# recalculée À LA MAIN ci-dessous.
sys.path.insert(0, os.path.join(RACINE, "tests"))
import lot_a_materiau as LA  # noqa: E402

with open("tests/ref_doubles_cornieres.json", encoding="utf-8") as fh:
    REF_SITE = json.load(fh)


def ecarts_site():
    """Les différences entre le moteur (table du site) et les trois notes figées."""
    out = []
    for nom, cas in REF_SITE.items():
        Rr = moteur.compute(cas["inp"])
        if {c.key for c in Rr.checks if c.active} != set(cas["checks"]):
            out.append(f"{nom} : vérifications actives divergentes")
        for k, (ed, rd, eta) in cas["checks"].items():
            c = Rr.ck.get(k)
            if c is None or not c.active or abs(c.Ed - ed) > 1e-9 * max(1, abs(ed)) \
               or abs(c.Rd - rd) > 1e-9 * max(1, abs(rd)) or abs(c.eta - eta) > 1e-12 * max(1, abs(eta)):
                out.append(f"{nom} {k} : attendu {(ed, rd, eta)}, obtenu {(c.Ed, c.Rd, c.eta) if c and c.active else 'inactif'}")
        for k, v in cas["scal"].items():
            if abs(Rr[k] - v) > 1e-9 * max(1, abs(v)):
                out.append(f"{nom} scalaire {k} : attendu {v}, obtenu {Rr[k]}")
        if (Rr.gov.key if Rr.gov else None) != cas["gov"] or bool(Rr.verified) != cas["verified"] \
           or sorted(a.id for a in Rr.alerts) != cas["alerts"]:
            out.append(f"{nom} : synthèse divergente")
    return out


def a_la_main():
    """Une ligne par cas, recalculée à la main (EN 1993-1-8 ; tableau 3.3 ANB) : (libellé, moteur, main)."""
    S3, gM2 = 3 ** 0.5, 1.25
    neuf = moteur.compute(REF_SITE["note_neuve"]["inp"])
    ipe = moteur.compute(REF_SITE["ipe600_heb600"]["inp"])
    s275 = moteur.compute(REF_SITE["s275_grugee"]["inp"])
    return [
        # 2 cornières L100x100x10, L = 190 mm, 3 trous de 22 mm, f_u = 470 : 2·t·(L − 3·d0)·f_u/(√3·γM2)
        ("note neuve — cornières B, cisaillement section nette : 2·10·(190 − 3·22)·470/(√3·1,25) = 538,37 kN",
         neuf.ck["cnB"].Rd, 2 * 10 * (190 - 3 * 22) * 470 / (S3 * gM2) / 1000),
        # âme de la porteuse HEA 400 (tf = 19 > 16 : f_y = 345), Av = 2 906,75 mm² : Av·f_y/(√3·γM0)
        ("note neuve — âme porteuse, cisaillement local : 2 906,75·345/√3 = 578,98 kN",
         neuf.ck["vlP"].Rd, 2906.75 * 345 / S3 / 1000),
        # IPE 600 (tf = 19 > 16 : f_y = 345), Av = 6 942 mm²
        ("IPE 600 sur HEB 600 — âme portée, cisaillement brut : 6 942·345/√3 = 1 382,75 kN",
         ipe.ck["vgS"].Rd, 6942 * 345 / S3 / 1000),
        # S275 : cornières de 10 mm f_u = 410 ; IPE 600 grugée (tf = 19 : f_y = 265), Av = 7 062 mm²
        ("S275 grugée — cornières B, section nette : 2·10·(190 − 3·22)·410/(√3·1,25) = 469,64 kN",
         s275.ck["cnB"].Rd, 2 * 10 * (190 - 3 * 22) * 410 / (S3 * gM2) / 1000),
        ("S275 grugée — âme portée, cisaillement brut : 7 062·265/√3 = 1 080,47 kN",
         s275.ck["vgS"].Rd, 7062 * 265 / S3 / 1000),
    ]


chk(f"les {len(REF_SITE)} notes figées sur la table du site sont reproduites", not ecarts_site(), " || ".join(ecarts_site()[:4]))
for lib, v, main in a_la_main():
    chk(lib, abs(v - main) <= 1e-9 * main, f"{v} / {main}")
_g = moteur.compute(REF_SITE["s275_grugee"]["inp"])
chk("S275 grugée : f_y = 265 pour la résistance, domaine lu sur la nuance (« Valide », limites S275 : 54,3)",
    _g.fy_S == 265 and _g.val_N == "Valide" and _g.lim_N == 54.3, (_g.fy_S, _g.val_N, _g.lim_N))

# mutation : l'ANCIENNE table (EN 1993-1-1 tableau 3.1, sans épaisseur) — les cas figés et les lignes à la main rougissent
LA._mut_table(True)
try:
    rouges_figes = len(ecarts_site())
    rouges_main = sum(1 for _, v, main in a_la_main() if abs(v - main) > 1e-9 * main)
finally:
    LA._mut_table(False)
chk(f"mutation « ancienne table » : les cas figés rougissent ({rouges_figes}) et les lignes à la main aussi ({rouges_main})",
    rouges_figes > 0 and rouges_main == 5)

print()
if diffs:
    print(f"RÉGRESSION CALCUL : {len(diffs)} différence(s)")
    for d in diffs[:40]:
        print("   -", d)
else:
    print("RÉGRESSION CALCUL : 0 différence")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:600])
sys.exit(1 if KO else 0)

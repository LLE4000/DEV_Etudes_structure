# -*- coding: utf-8 -*-
"""Le catalogue livré ``acier/donnees/profiles.json`` et son chargeur
``acier/profiles/catalogue.py`` : contenu (familles, nombres, champs),
cohérence avec le moteur (un échantillon recalculé donne les mêmes
nombres), avec les données que l'application portait AVANT le catalogue
(``tests/ref_profiles_legacy.json`` = l'ancien profiles_test.json, 48
profilés) et avec les dimensions de la bibliothèque des assemblages
(``acier/donnees/bibliotheques.json``, 90 profilés du corrigé de
référence) ; filtres, tri, recherche, préférences, forme ancienne.

Lancement : python3 tests/test_profiles_catalogue.py
"""
import json
import os
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.profiles import catalogue as CAT  # noqa: E402
from acier.profiles import familles as F  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def ecart(a, b):
    return abs(a / b - 1.0) if b else abs(a - b)


print("=== 1. Contenu ===")
P = CAT.profils()
fam = dict(CAT.familles_presentes())
chk("au moins 1 400 profilés, 21 familles, méta datée avec sources et contrôle",
    len(P) >= 1400 and len(fam) == 21 and all(k in CAT.meta() for k in ("date", "sources_dimensions", "controle", "calcul")), str(len(P)))
attendus = {"IPE": 18, "IPN": 21, "HEA": 24, "HEB": 24, "HEM": 24, "HEAA": 24, "HL": 35, "HD": 38, "HP": 31, "UPE": 14, "UPN": 18,
            "L": 80, "LI": 15, "CHS": 160, "SHS": 120, "RHS": 120, "CHS-F": 100, "SHS-F": 90, "RHS-F": 130}
chk("nombres par famille au moins ceux des listes normalisées", all(fam.get(f, 0) >= n for f, n in attendus.items()),
    str({f: fam.get(f, 0) for f in attendus if fam.get(f, 0) < attendus[f]}))
noms = [p["nom"] for p in P]
chk("noms uniques, famille connue, dimensions de la famille présentes et positives",
    len(set(noms)) == len(noms) and all(p["famille"] in F.FAMILLES for p in P)
    and all(all(float(p["dims"][k]) > 0 for k in F.FAMILLES[p["famille"]]["dims"]) for p in P))
cles = ("A", "masse", "Iy", "Wel_y", "Wpl_y", "iy", "Iz", "Wel_z", "Wpl_z", "iz", "It", "perimetre", "cf_tf")
chk("propriétés principales présentes et positives sur tous les profilés",
    all(p.get(k) is not None and p[k] > 0 for p in P for k in cles), str([p["nom"] for p in P if any(p.get(k) is None or p[k] <= 0 for k in cles)][:5]))
chk("I/H et U : Avz, Avy, Iw, d_ame (I/H) présents ; cornières : Iu > Iy > Iz > Iv ou égales, α ; creux : Wt, Iw = 0",
    all(p.get("Avz") and p.get("Avy") and p.get("Iw") is not None for p in P if F.FAMILLES[p["famille"]]["groupe"] in ("I/H", "U"))
    and all(p.get("d_ame") for p in P if F.FAMILLES[p["famille"]]["groupe"] == "I/H")
    and all(p["Iu"] >= p["Iy"] >= p["Iz"] >= p["Iv"] and p.get("alpha_deg") is not None for p in P if F.FAMILLES[p["famille"]]["groupe"] == "L")
    and all(p.get("Wt") and p.get("Iw") == 0 for p in P if F.FAMILLES[p["famille"]]["groupe"] == "creux"))
chk("cornières égales : α = 45° ; inégales : 0 < α < 45°",
    all(abs(p["alpha_deg"] - 45.0) < 1e-6 for p in P if p["famille"] == "L")
    and all(0.0 < p["alpha_deg"] < 45.0 for p in P if p["famille"] == "LI"))
chk("masses croissantes avec la hauteur dans IPE, HEA, HEB, HEM, UPE, UPN",
    all(all(a["masse"] < b["masse"] for a, b in zip(l, l[1:])) for l in
        ([p for p in P if p["famille"] == f and "x" not in p["nom"].split()[-1]] for f in ("IPE", "HEA", "HEB", "HEM", "UPE", "UPN"))))
chk("séries courantes : IPE, HE A/B/M, UPE, UPN, L EN 10056, creux ; non courantes : IPE A/O, IPN, HE AA, HL, HD, HP, cornières étendues",
    all(p["courant"] for p in P if p["famille"] in ("IPE", "HEA", "HEB", "HEM", "UPE", "UPN", "CHS", "SHS", "RHS"))
    and not any(p["courant"] for p in P if p["famille"] in ("IPE A", "IPE O", "IPN", "HEAA", "HL", "HD", "HP"))
    and any(not p["courant"] for p in P if p["famille"] == "L") and any(p["courant"] for p in P if p["famille"] == "L"))
chk("HL 1100 : rayon déduit de la masse publiée, dit dans la fiche (note)",
    all(p.get("note") and p["dims"]["r"] == 20.0 for p in P if p["nom"].startswith("HL 1100")))

print("\n=== 2. Le fichier dit ce que le moteur calcule ===")
import random  # noqa: E402
rng = random.Random(4)
ech = rng.sample(P, 24)
pires = []
for p in ech:
    q = F.calculer(p["famille"], p["dims"], torsion_exacte=False)
    for k in ("A", "Iy", "Iz", "Wel_y", "Wpl_y", "Wpl_z", "iy", "iz", "masse", "perimetre", "Iw", "Wt", "Avz", "cf_tf"):
        if p.get(k) is not None and q.get(k) is not None and p[k] != 0:
            pires.append((ecart(q[k], p[k]), p["nom"], k))
pires.sort(reverse=True)
chk("24 profilés tirés au sort recalculés : mêmes nombres à 10⁻⁵ (arrondi à 6 chiffres du fichier)", pires[0][0] < 1e-5, str(pires[:3]))

print("\n=== 3. Les données d'avant (profiles_test.json, 48 profilés) sont retrouvées ===")
with open(os.path.join("tests", "ref_profiles_legacy.json"), encoding="utf-8") as fh:
    LEG = json.load(fh)
par = CAT.par_nom()
# l'ancien fichier nommait « IPEA 200 » ce que le catalogue nomme « IPE A 200 »
LEG = {(n.replace("IPEA ", "IPE A ") if n.startswith("IPEA ") else n): v for n, v in LEG.items()}
corr = {"masse": "masse", "A": "A", "Iy": "Iy", "Wel": "Wel_y", "Wpl_y": "Wpl_y", "Avz": "Avz", "Iz": "Iz",
        "Wel_z": "Wel_z", "Wpl_z": "Wpl_z"}
absents = [n for n in LEG if n not in par]
chk("les 48 profilés d'avant existent dans le catalogue (IPE, IPE A, HEA, HEB, HEM)", not absents, str(absents))
pires, pires_i = [], []
for n, v in LEG.items():
    if n not in par:
        continue
    for k, kc in corr.items():
        if v.get(k) and par[n].get(kc):
            pires.append((ecart(par[n][kc], float(v[k])), n, k))
    for k in ("iy", "iz"):                      # l'ancien fichier arrondissait i à une décimale (IPE 80 : iz « 1,0 » pour 1,05)
        if v.get(k) and par[n].get(k):
            pires_i.append((abs(par[n][k] - float(v[k])), n, k))
pires.sort(reverse=True)
pires_i.sort(reverse=True)
chk("dimensions h, b, tw, tf, r identiques", all(abs(float(v[k]) - par[n]["dims"][k]) < 1e-9 for n, v in LEG.items() if n in par for k in ("h", "b", "tw", "tf", "r")))
chk("masse, A, Iy, Wel, Wpl_y, Avz, Iz, Wel_z, Wpl_z à 1 % des valeurs d'avant (arrondis de l'ancien fichier)",
    pires[0][0] < 0.01, str(pires[:3]))
chk("iy, iz à 0,11 cm près (l'ancien fichier les arrondissait à une décimale : ± 0,05, plus l'écart publié / calculé)",
    pires_i[0][0] <= 0.11, str(pires_i[:3]))

print("\n=== 4. La bibliothèque des assemblages (corrigé de référence) a les mêmes dimensions ===")
with open(os.path.join("acier", "donnees", "bibliotheques.json"), encoding="utf-8") as fh:
    DB = json.load(fh)
communs = [x for x in DB["profils"] if x["n"] in par]
div = [x["n"] for x in communs if any(abs(float(x[k]) - par[x["n"]]["dims"][k]) > 1e-9 for k in ("h", "b", "tw", "tf", "r"))]
chk("au moins 80 des 90 profilés du corrigé sont au catalogue, dimensions h, b, tw, tf, r identiques",
    len(communs) >= 80 and not div, f"{len(communs)} communs ; divergents : {div}")

print("\n=== 5. Chargeur : recherche, filtres, tri, forme ancienne, préférences ===")
chk("recherche insensible aux espaces et à la casse : « ipe200 » → IPE 200 (+ IPE A 200, IPE O 200)",
    {p["nom"] for p in CAT.filtrer(P, None, "ipe200")} >= {"IPE 200"} and all("200" in p["nom"] for p in CAT.filtrer(P, None, "ipe200")))
chk("recherche « 100x100 » : tubes carrés et cornières 100 × 100 seulement", all("100x100" in p["nom"] for p in CAT.filtrer(P, None, "100x100"))
    and {p["famille"] for p in CAT.filtrer(P, None, "100x100")} >= {"L", "SHS"})
f1 = CAT.filtrer(P, ["IPE", "HEA"], "", {"Iy": (1500.0, None), "masse": (None, 40.0)})
chk("filtres : IPE et HEA avec Iy ≥ 1 500 et masse ≤ 40 → liste non vide, bornes tenues, familles tenues",
    f1 and all(p["Iy"] >= 1500 and p["masse"] <= 40 and p["famille"] in ("IPE", "HEA") for p in f1))
chk("un profilé sans la propriété filtrée est exclu (cornières sans Iw)", not CAT.filtrer(P, ["L"], "", {"Iw": (0.0, None)}))
tri = CAT.trier(f1, "Iy", True)
chk("tri décroissant par Iy", all(a["Iy"] >= b["Iy"] for a, b in zip(tri, tri[1:])))
chk("courants seulement : plus d'IPE A ni de HL", not any(p["famille"] in ("IPE A", "HL") for p in CAT.filtrer(P, None, "", None, True)))
leg = CAT.legacy(("IPE", "HEA"))
chk("forme ancienne : clés de profiles_test.json, type = famille, hi = h − 2tf, d = d_ame",
    all(k in leg["IPE 200"] for k in ("masse", "h", "b", "tw", "tf", "r", "hi", "d", "A", "Iy", "Wel", "Wpl_y", "iy", "Avz", "Iz", "Wel_z", "Wpl_z", "iz", "type"))
    and leg["IPE 200"]["type"] == "IPE" and abs(leg["IPE 200"]["hi"] - 183.0) < 1e-9 and leg["IPE 200"]["d"] == par["IPE 200"]["d_ame"]
    and "HEB 200" not in leg)
chk("colonnes : chaque colonne a libellé, unité, groupe, décimales ; les défauts existent",
    all(c in CAT.LIBELLE and c in CAT.GROUPE_DE for c in CAT.DEFAUT_COLONNES) and set(CAT.GROUPE_DE.values()) == set(CAT.GROUPES_COLONNES))
chk("formatage français : 1 943,4 ; Iw grand en ×10³ ; absent → —",
    CAT.formater(1943.41, "Iy") == "1 943,4" and "×10³" in CAT.formater(125000.0, "Iw") and CAT.formater(None, "Iy") == "—")
# préférences : fichier temporaire pour ne pas toucher au fichier livré
orig = CAT.CHEMIN_PREFS
try:
    with tempfile.TemporaryDirectory() as d:
        CAT.CHEMIN_PREFS = os.path.join(d, "prefs.json")
        p0 = CAT.preferences()
        CAT.enregistrer_preferences({**p0, "colonnes": ["h", "Iy", "truc"], "familles": ["IPE", "XXX"], "tri": "masse"})
        p1 = CAT.preferences()
        chk("préférences : enregistrées puis relues, colonnes et familles inconnues écartées",
            p1["colonnes"] == ["h", "Iy"] and p1["familles"] == ["IPE"] and p1["tri"] == "masse")
        CAT.retablir_preferences()
        chk("rétablir : retour aux défauts", CAT.preferences()["colonnes"] == CAT.DEFAUT_COLONNES)
finally:
    CAT.CHEMIN_PREFS = orig
chk("le fichier de préférences livré est lisible et porte les défauts du bureau", isinstance(CAT.preferences()["colonnes"], list))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

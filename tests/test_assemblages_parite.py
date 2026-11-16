# -*- coding: utf-8 -*-
"""Parité du moteur Python avec le moteur HTML de référence.

Corrigé : acier/reference/reference_double_corniere.json (produit par le
moteur du HTML via l'oracle Node). Chaque garantie rougit si on la retire :
  1. LES 31 CAS : chaque scalaire, chaque vérification (active, Ed, Rd, η,
     ok, vals), chaque alerte (identifiant, bloquante, message, explication,
     champs, cotes, éléments), chaque ligne du Tableau 3.3, les efforts par
     boulon, le prédimensionnement (solution retenue, et toutes les lignes
     pour « defaut » et « predim_* »), le statut, le taux maximal et la
     vérification gouvernante — écart relatif ≤ 1e-9, égalité stricte sur
     les booléens, identifiants, chaînes.
  2. LE BENCHMARK : les 37 valeurs publiées et les 22 valeurs de calcul
     manuel, avec des statuts identiques ; statut global VALIDÉ.
  3. L'INSTANTANÉ a exactement la forme du corrigé (mêmes clés, mêmes
     comptes : 190 scalaires, 38 vérifications, 8 lignes du Tableau 3.3 sur
     le cas par défaut).
  4. L'ÉCART NOMMÉ (lot A, 16/11/2026) : le corrigé reste figé sur SON
     matériau (EN 1993-1-1 tableau 3.1, sans épaisseur), que la parité lui
     donne (``parite.materiau_du_corrige``) — une validation de FORMULES.
     Sans lui, le moteur prend la table belge (NBN EN 1993-1-1 ANB:2018
     3.2.1(1), tableau 3.3 ANB) : le cas par défaut s'en écarte par f_y de la
     porteuse (HEB 300, t = 19 mm : 345) et f_u (470), et par rien d'autre
     que ce qui en découle.

Lancement : python3 tests/test_assemblages_parite.py (depuis la racine).
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages.poutre_poutre.doubles_cornieres import moteur, benchmark  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres.parite import (  # noqa: E402
    instantane, comparer, materiau_du_corrige)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


with open(os.path.join(RACINE, "acier", "reference", "reference_double_corniere.json"),
          encoding="utf-8") as fh:
    REF = json.load(fh)
CORRIGE = materiau_du_corrige(REF)          # le matériau du corrigé (écart nommé avec la table belge, voir 4.)

# ================================================================
print("=== 1. Les 31 cas du corrigé ===")
total = 0
for cas in REF["cas"]:
    attendu = cas["attendu"]
    try:
        R = moteur.compute(cas["inputs"], CORRIGE)
        obtenu = instantane(R, lignes_predim="lignes" in attendu["predim"])
        n, ecarts = comparer(attendu, obtenu, tol=1e-9, chemin=cas["id"])
    except Exception as e:  # noqa: BLE001 — un plantage est un échec de parité
        n, ecarts = 0, [f"{cas['id']} : exception {type(e).__name__}: {e}"]
    total += n
    chk(f"{cas['id']} — {n} grandeurs, {len(ecarts)} écart(s)", not ecarts,
        " | ".join(ecarts[:4]))
chk(f"au moins 23 000 grandeurs comparées sur les 31 cas ({total})", total >= 23000)

# ================================================================
print("\n=== 2. Le benchmark du module ===")
B = benchmark.run_bench()
BR = REF["benchmark"]
chk("statut global VALIDÉ, identique au corrigé",
    B["statut"] == BR["statut"] == "VALIDÉ" and B["ok"] is True)
chk("écart maximal identique", abs(B["worst"] - BR["worst"]) <= 1e-9 * max(1, BR["worst"]),
    f"{B['worst']} / {BR['worst']}")
n_pub = 0
ec = []
for b, br in zip(B["bench"], BR["bench"]):
    for r, rr in zip(b["rows"], br["rows"]):
        n_pub += 1
        if abs(r["val"] - rr["val"]) > 1e-9 * max(1, abs(rr["val"])) or r["st"] != rr["st"] \
                or abs(r["ec"] - rr["ec"]) > 1e-9:
            ec.append(f"{b['id']} {rr['lab'][:40]} : {r['val']} / {rr['val']} ; {r['st']} / {rr['st']}")
chk(f"37 valeurs publiées : valeurs et statuts identiques ({n_pub})", n_pub == 37 and not ec,
    " | ".join(ec[:3]))
n_man = 0
ec = []
for c, cr in zip(B["valid"], BR["valid"]):
    for r, rr in zip(c["rows"], cr["rows"]):
        n_man += 1
        if abs(r["val"] - rr["val"]) > 1e-9 * max(1, abs(rr["val"])) or r["st"] != rr["st"]:
            ec.append(f"{c['id']} {rr['lab'][:40]} : {r['val']} / {rr['val']}")
chk(f"22 valeurs de calcul manuel : valeurs et statuts identiques ({n_man})",
    n_man == 22 and not ec, " | ".join(ec[:3]))
chk("toutes les lignes manuelles sont OK",
    all(r["st"] == "OK" for c in B["valid"] for r in c["rows"]))

# ================================================================
print("\n=== 3. Forme de l'instantané ===")
R0 = moteur.compute({}, CORRIGE)
S0 = instantane(R0)
A0 = REF["cas"][0]["attendu"]
chk("mêmes clés de premier niveau que le corrigé", set(S0) == set(A0))
chk("190 scalaires sur le cas par défaut", len(S0["scalaires"]) == 190, str(len(S0["scalaires"])))
chk("38 vérifications, dans l'ordre du corrigé",
    [c["key"] for c in S0["verifications"]] == [c["key"] for c in A0["verifications"]])
chk("8 lignes du Tableau 3.3 sur le cas par défaut", len(S0["tableau_3_3"]) == 8)
chk("56 lignes de prédimensionnement (8 boulons × 7 nombres de rangées)",
    len(S0["predim"]["lignes"]) == 56)
chk("le catalogue des vérifications (libellé, unité, référence, nature, formule, "
    "essentielle) est celui du corrigé",
    [(c.key, c.grp, c.label, c.unit, c.ref, c.nat, c.formula, c.ess) for c in R0.checks]
    == [(c["key"], c["groupe"], c["libelle"], c["unite"], c["reference"], c["nature"],
         c["formule"], c["essentielle"]) for c in REF["catalogue_verifications"]])
chk("valeurs par défaut : les 84 clés du corrigé",
    moteur.defaults() == REF["valeurs_par_defaut"])

# ================================================================
print("\n=== 4. L'écart nommé : la table belge, sans le matériau du corrigé ===")
Rb = moteur.compute({})
nb, ecb = comparer(REF["cas"][0]["attendu"], instantane(Rb), tol=1e-9, chemin="defaut")
mat = sorted(e.split(" : ")[0].rsplit(".", 1)[1] for e in ecb if ".scalaires.f" in e and e.split(" : ")[0].rsplit(".", 1)[1][:3] in ("fy_", "fu_"))
chk("sans le matériau du corrigé, le cas par défaut s'en écarte (table belge)", bool(ecb), str(len(ecb)))
chk("… par f_y de la porteuse (HEB 300, t = 19 mm : 345) et f_u des trois pièces (470) — tableau 3.3 ANB",
    mat == ["fu_C", "fu_P", "fu_S", "fy_P"] and Rb.fy_P == 345 and Rb.fy_S == 355 and Rb.fy_C == 355
    and Rb.fu_P == Rb.fu_S == Rb.fu_C == 470, str(mat))
chk("… et le taux gouvernant suit f_u : 0,7399 × 510/470 = 0,8029 (pression diamétrale de l'âme portée)",
    Rb.gov.key == "pdS" and abs(Rb.eta_max - R0.eta_max * 510 / 470) < 1e-12, f"{Rb.eta_max} / {R0.eta_max}")

# ================================================================
print("\n=== 5. Lot A : f_y et f_u de chaque pièce selon SON épaisseur (tableau 3.3 ANB) ===")
# NBN EN 1993-1-1 ANB:2018 3.2.1(1) (option a) ; profilé : max(tf, tw) ; cornières : leur épaisseur.
sys.path.insert(0, os.path.join(RACINE, "tests"))
import lot_a_materiau as LA  # noqa: E402
from acier.bibliotheques import PERSO  # noqa: E402
LA.jouer(chk, moteur.compute, attendus=[
    ("note neuve", {}, {"P": (19, LA.FY_S355_40, LA.FU_S355), "S": (14, LA.FY_S355, LA.FU_S355),
                        "C": (10, LA.FY_S355, LA.FU_S355)}),
    ("IPE 600 portée par un HEB 600 (tf = 19 et 30 mm)", dict(prof_S="IPE 600", prof_P="HEB 600", V_Ed=250),
     {"S": (19, LA.FY_S355_40, LA.FU_S355), "P": (30, LA.FY_S355_40, LA.FU_S355)}),
    ("cornières de 15 mm (L150x150x15)", dict(corn_u="L150x150x15"), {"C": (15, LA.FY_S355, LA.FU_S355)}),
    ("S460 N/NL (tableau 3.4 ANB)", dict(nu_S="S460 N/NL", prof_S="IPE 600"), {"S": (19, LA.FY_S460_40, LA.FU_S460)}),
], ligne_fu=("note neuve", {}, "pdS"),
   grugeage=(dict(prof_S="IPE 600", prof_P="HEB 600", nu_S="S275", d_nt=40, l_n=120, V_Ed=250), "S"),
   perso=(dict(nu_C=PERSO, fy_u=300, fu_u=420, bw_u=0.85), "C", 300, 420, "cornières, f_y et f_u saisis tels quels"),
   hors_table=(dict(prof_P=PERSO, hP_u=800, bP_u=300, twP_u=15, tfP_u=300, rP_u=27), "poutre principale"))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)

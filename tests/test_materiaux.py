# -*- coding: utf-8 -*-
"""LA table des aciers de construction (acier/materiaux.py) : f_y et f_u selon l'épaisseur, option a de la
NBN EN 1993-1-1 ANB:2018 (3.2.1(1) ; tableaux 3.3 ANB à 3.12 ANB).

  1. Les VALEURS de chaque ligne reprise, bande par bande, aux bornes et au milieu des bandes : elles sont écrites ici
     telles que deux lectures indépendantes du PDF de l'ANB les ont transcrites le 16/11/2026 (concordantes, 88 lignes
     sur 88) — pas recopiées du module.
  2. Les cas du brief (test 25) : HEB 200 S235 (t_f 15) → 235 ; HEB 300 S235 (t_f 19) → 225 ; plat de 20 mm en S235 →
     225 ; plats de 30 mm : S235 → 225, S355 → 345. f_u : S355 470, S275 410 entre 3 et 100 mm (510 et 430 sous 3 mm).
  3. La FORME : bandes contiguës et croissantes, f_y et f_u ne croissent jamais avec l'épaisseur, chaque nuance et
     chaque produit résolus, au-delà du tableau un refus nommé (aucune valeur inventée).
  4. Les noms des notes enregistrées (« S355 (fu 490) », « S355 N/NL », « S460 N/NL ») suivent la table belge.
  5. La référence imprimée (« tableau 3.3 ANB (EN 10025-2:2004, S355JR), 16 < t ≤ 40 mm »).
  6. SOURCE UNIQUE : acier/flambement/normes.limite_elastique rend la valeur de la table (bornes du module en plus) ;
     la bibliothèque des assemblages ne porte plus de f_y ni de f_u.
  7. Mutations : chaque altération plausible de la table (l'ancien palier à 40 mm, f_u = 510 du tableau 3.1, une borne
     « < 3 » lue « ≤ 3 », un alias perdu, un produit confondu…) fait rougir au moins un contrôle.
  8. Pureté : la table se charge sans rien d'autre que la bibliothèque standard.

    /usr/bin/python3.12 tests/test_materiaux.py
"""
import copy
import json
import os
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier import materiaux as MA  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info and not cond else ""))


def refus(f, *a):
    try:
        f(*a)
    except MA.HorsTable:
        return True
    return False


# ------------------------------------------------------------------ 1. les valeurs, telles que transcrites
# (nuance, produit) → [(t, f_y attendu)] et [(t, f_u attendu)] ; None = refus (hors du tableau)
FY = {
    ("S235", "lamine"): [(1, 235), (16, 235), (16.5, 225), (40, 225), (40.5, 215), (63, 215), (80, 215), (100, 215),
                         (100.5, 195), (150, 195), (151, 185), (200, 185), (201, 175), (250, 175), (251, None)],
    ("S275", "lamine"): [(16, 275), (17, 265), (40, 265), (41, 255), (63, 255), (64, 245), (80, 245), (81, 235),
                         (100, 235), (101, 225), (150, 225), (151, 215), (200, 215), (201, 205), (250, 205), (251, None)],
    ("S355", "lamine"): [(16, 355), (16.1, 345), (40, 345), (41, 335), (63, 335), (64, 325), (80, 325), (81, 315),
                         (100, 315), (101, 295), (150, 295), (151, 285), (200, 285), (201, 275), (250, 275), (251, None)],
    ("S460", "lamine"): [(16, 460), (17, 440), (40, 440), (41, 430), (63, 430), (64, 410), (80, 410), (81, 400),
                         (100, 400), (101, 380), (150, 380), (151, 370), (200, 370), (201, None)],
    ("S355 N/NL", "lamine"): [(16, 355), (17, 345), (63, 335), (80, 325), (100, 315), (150, 295), (200, 285),
                              (250, 275), (251, None)],
    ("S235", "creux_chaud"): [(16, 235), (17, 225), (40, 225), (41, 215), (100, 215), (101, 195), (120, 195), (121, None)],
    ("S275", "creux_chaud"): [(16, 275), (17, 265), (41, 255), (64, 245), (81, 235), (101, 225), (120, 225), (121, None)],
    ("S355", "creux_chaud"): [(16, 355), (17, 345), (41, 335), (64, 325), (81, 315), (101, 295), (120, 295), (121, None)],
    ("S460", "creux_chaud"): [(16, 460), (17, 440), (40, 440), (41, 430), (65, 430), (66, None)],
    ("S355 N/NL", "creux_chaud"): [(16, 355), (17, 345), (41, 335), (65, 335), (66, None)],
    ("S235", "creux_froid"): [(16, 235), (17, 225), (40, 225), (41, None)],
    ("S275", "creux_froid"): [(16, 275), (17, 265), (40, 265), (41, None)],
    ("S355", "creux_froid"): [(16, 355), (17, 345), (40, 345), (41, None)],
    ("S460", "creux_froid"): [(16, 460), (17, 440), (40, 440), (41, None)],
}
FU = {
    ("S235", "lamine"): [(2, 360), (2.99, 360), (3, 360), (100, 360), (101, 350), (150, 350), (151, 340), (250, 340),
                         (251, None)],
    ("S275", "lamine"): [(2, 430), (2.99, 430), (3, 410), (100, 410), (101, 400), (150, 400), (151, 380), (250, 380)],
    ("S355", "lamine"): [(2, 510), (2.99, 510), (3, 470), (40, 470), (100, 470), (101, 450), (250, 450), (251, None)],
    ("S460", "lamine"): [(2, 540), (100, 540), (101, 530), (200, 530), (201, None)],
    ("S355 N/NL", "lamine"): [(2, 470), (100, 470), (101, 450), (250, 450)],
    ("S235", "creux_chaud"): [(2, 360), (3, 360), (100, 360), (101, 350), (120, 350), (121, None)],
    ("S275", "creux_chaud"): [(2, 430), (3, 430), (3.2, 410), (100, 410), (101, 400)],
    ("S355", "creux_chaud"): [(2, 510), (3, 510), (3.2, 470), (100, 470), (101, 450), (120, 450)],
    ("S460", "creux_chaud"): [(2, 540), (65, 540), (66, None)],
    ("S235", "creux_froid"): [(2, 360), (3, 360), (40, 360), (41, None)],
    ("S275", "creux_froid"): [(2, 430), (2.99, 430), (3, 410), (40, 410)],
    ("S355", "creux_froid"): [(2, 510), (2.99, 510), (3, 470), (40, 470), (41, None)],
    ("S460", "creux_froid"): [(2, 540), (40, 540)],
}


def controles(M=MA):
    """[(nom, condition, info)] : la même liste sert au rapport et aux mutations."""
    C = []

    def c(nom, cond, info=""):
        C.append((nom, bool(cond), info))

    def lu(f, nuance, t, produit):
        try:
            return f(nuance, t, produit)
        except M.HorsTable:
            return None

    for (n, p), cas in FY.items():
        mauvais = [(t, v, lu(M.fy, n, t, p)) for t, v in cas if lu(M.fy, n, t, p) != v]
        c(f"f_y {n} {p} : {len(cas)} épaisseurs, chaque bande et ses bornes", not mauvais, mauvais)
    for (n, p), cas in FU.items():
        mauvais = [(t, v, lu(M.fu, n, t, p)) for t, v in cas if lu(M.fu, n, t, p) != v]
        c(f"f_u {n} {p} : {len(cas)} épaisseurs, chaque bande et ses bornes", not mauvais, mauvais)
    # 2. les cas du brief (test 25)
    c("brief 25 : HEB 200 S235 (t_f 15) → 235 ; HEB 300 S235 (t_f 19) → 225",
      lu(M.fy, "S235", 15.0, "lamine") == 235 and lu(M.fy, "S235", 19.0, "lamine") == 225)
    c("brief 25 : plat de 20 mm S235 → 225 ; plats de 30 mm : S235 → 225, S355 → 345",
      lu(M.fy, "S235", 20, "lamine") == 225 and lu(M.fy, "S235", 30, "lamine") == 225 and lu(M.fy, "S355", 30, "lamine") == 345)
    c("f_u entre 3 et 100 mm : S235 360, S275 410, S355 470 (tableau 3.1 de l'EN : 360 / 430 / 510 — non admis)",
      [lu(M.fu, n, 10, "lamine") for n in ("S235", "S275", "S355")] == [360, 410, 470])
    # 4. les noms des notes enregistrées
    c("« S355 (fu 490) » suit le S355 belge : f_y 345 à 20 mm, f_u 470 (aucun 490 dans les tableaux de l'ANB)",
      lu(M.fy, "S355 (fu 490)", 20, "lamine") == 345 and lu(M.fu, "S355 (fu 490)", 10, "lamine") == 470)
    c("« S460 N/NL » = tableau 3.4 ANB (S460N) : f_y 440 de 16 à 40 mm, f_u 540",
      lu(M.fy, "S460 N/NL", 20, "lamine") == 440 and lu(M.fu, "S460 N/NL", 20, "lamine") == 540)
    # 5. la référence
    c("source : « tableau 3.3 ANB (EN 10025-2:2004, S355JR), 16 < t ≤ 40 mm »",
      M.source("S355", 20) == "tableau 3.3 ANB (EN 10025-2:2004, S355JR), 16 < t ≤ 40 mm", M.source("S355", 20))
    c("source de f_u, tableau 3.3 ANB : « 3 ≤ t ≤ 100 mm » ; tableau 3.8 ANB : « 3 < t ≤ 100 mm »",
      M.source("S355", 10, "lamine", "fu").endswith("3 ≤ t ≤ 100 mm")
      and M.source("S355", 10, "creux_chaud", "fu").endswith("3 < t ≤ 100 mm"),
      (M.source("S355", 10, "lamine", "fu"), M.source("S355", 10, "creux_chaud", "fu")))
    c("source d'un tube formé à froid : tableau 3.10 ANB (EN 10219-1:2006, S355J0H)",
      M.source("S355", 8, "creux_froid").startswith("tableau 3.10 ANB (EN 10219-1:2006, S355J0H)"), M.source("S355", 8, "creux_froid"))
    c("nominale : le nombre du nom (235, 275, 355, 460)", [M.nominale(n) for n in M.NUANCES] == [235, 275, 355, 460])
    return C


# ------------------------------------------------------------------ 3. la forme
def forme():
    C = []
    for tab, (norme, lignes) in MA.TABLEAUX.items():
        for des, (reh, rm) in lignes.items():
            for nom, bandes in (("R_eH", reh), ("R_m", rm)):
                sups = [b[0] for b in bandes]
                vals = [b[2] for b in bandes]
                C.append((f"tableau {tab} ANB {des} {nom} : bornes croissantes, valeurs qui ne croissent pas avec t",
                          sups == sorted(sups) and len(set(sups)) == len(sups) and vals == sorted(vals, reverse=True), bandes))
    tous = all(isinstance(MA.fy(n, 10, p), float) and isinstance(MA.fu(n, 10, p), float)
               for n in list(MA.NUANCES) + list(MA.ALIAS) for p in MA.PRODUITS)
    C.append(("chaque nuance du site et chaque nom enregistré, pour chaque produit, se lit dans un tableau", tous))
    C.append(("une nuance inconnue, un produit inconnu, une épaisseur nulle ou négative : refus nommé",
              refus(MA.fy, "S420", 10) and refus(MA.fy, "S235", 10, "inox") and refus(MA.fy, "S235", 0)
              and refus(MA.fy, "S235", -5) and refus(MA.fu, "S235", float("nan"))))
    try:
        MA.fy("S355", 50, "creux_froid")
        msg = ""
    except MA.HorsTable as e:
        msg = str(e)
    C.append(("au-delà du tableau, le refus dit le tableau et sa borne (« tableau 3.10 ANB s'arrête à 40 mm »)",
              "3.10 ANB" in msg and "40 mm" in msg, msg))
    # le refus s'imprime tel quel à l'écran : le nom du produit, jamais sa clé interne ni « f_y : » en tête
    C.append(("le refus nomme le produit en français (« creux formé à froid »), sans clé interne ni « f_y : » en tête",
              msg.startswith("S355 creux formé à froid de 50 mm") and "creux_froid" not in msg
              and "(creux" not in msg and not msg.startswith("f_y"), msg))
    try:
        MA.fu("S235", 300)
        msg = ""
    except MA.HorsTable as e:
        msg = str(e)
    C.append(("refus d'un laminé : « S235 laminé de 300 mm : le tableau 3.3 ANB s'arrête à 250 mm (f_u) »",
              msg == "S235 laminé de 300 mm : le tableau 3.3 ANB s'arrête à 250 mm (f_u)", msg))
    C.append(("chaque produit a son libellé imprimable", set(MA.LIBELLES_PRODUIT) == set(MA.PRODUITS),
              MA.LIBELLES_PRODUIT))
    C.append(("paliers : la ligne entière, pour l'écran", MA.paliers("S235")[:2] == [("t ≤ 16 mm", 235.0), ("16 < t ≤ 40 mm", 225.0)],
              MA.paliers("S235")[:2]))
    return C


# ------------------------------------------------------------------ 6. source unique
def source_unique():
    C = []
    from acier.flambement import normes as NO
    egal = all(NO.limite_elastique(n, t, froid=f) == MA.fy(n, t, "creux_froid" if f else "lamine")
               for n in MA.NUANCES for t in (5, 16, 16.5, 30, 40, 41, 63, 64, 80) for f in (False,)) \
        and all(NO.limite_elastique(n, t, froid=True) == MA.fy(n, t, "creux_froid") for n in MA.NUANCES for t in (5, 16, 17, 40))
    C.append(("normes.limite_elastique = la table, pour chaque nuance et chaque palier (laminés et formés à froid)", egal))
    try:
        NO.limite_elastique("S355", 81)
        borne = False
    except NO.HorsDomaine:
        borne = True
    C.append(("normes.limite_elastique garde la borne de 80 mm des modules Flambement et Barres", borne))
    C.append(("normes.NUANCES : les nuances du site, avec leur valeur nominale", NO.NUANCES == {n: MA.nominale(n) for n in MA.NUANCES},
              NO.NUANCES))
    with open("acier/donnees/bibliotheques.json", encoding="utf-8") as fh:
        db = json.load(fh)
    C.append(("bibliotheques.json : ses aciers n'ont plus ni f_y ni f_u (la table les donne selon l'épaisseur)",
              all("fy" not in a and "fu" not in a for a in db["aciers"]), db["aciers"]))
    from acier import bibliotheques as BI
    a = BI.acier("S355", 20.0)
    C.append(("bibliotheques.acier(« S355 », 20 mm) : f_y 345, f_u 470 de la table, β_w 0,9 de la bibliothèque",
              (a["fy"], a["fu"], a["bw"]) == (345.0, 470.0, 0.9), a))
    return C


# ------------------------------------------------------------------ 7. mutations
def mutations():
    sauf = copy.deepcopy(MA.TABLEAUX)
    sauf_alias = dict(MA.ALIAS)
    sauf_lignes = copy.deepcopy(MA.LIGNES)

    def restaure():
        MA.TABLEAUX.clear()
        MA.TABLEAUX.update(copy.deepcopy(sauf))
        MA.ALIAS.clear()
        MA.ALIAS.update(sauf_alias)
        MA.LIGNES.clear()
        MA.LIGNES.update(copy.deepcopy(sauf_lignes))

    def remplace(tab, des, rang, bandes):
        norme, lignes = MA.TABLEAUX[tab]
        r = list(lignes[des])
        r[rang] = bandes
        lignes[des] = tuple(r)

    def option_b():          # l'ancien tableau 3.1 : un seul palier, à 40 mm
        for des, v in (("S235JR", 235), ("S275JR", 275), ("S355JR", 355)):
            reh = list(MA.TABLEAUX["3.3"][1][des][0])
            reh[1] = (40.0, True, float(v))
            remplace("3.3", des, 0, tuple(reh))

    def fu_510():
        rm = list(MA.TABLEAUX["3.3"][1]["S355JR"][1])
        rm[1] = (100.0, True, 510.0)
        remplace("3.3", "S355JR", 1, tuple(rm))

    def fu_430():
        rm = list(MA.TABLEAUX["3.3"][1]["S275JR"][1])
        rm[1] = (100.0, True, 430.0)
        remplace("3.3", "S275JR", 1, tuple(rm))

    def borne_3_incluse():
        rm = list(MA.TABLEAUX["3.3"][1]["S355JR"][1])
        rm[0] = (3.0, True, 510.0)
        remplace("3.3", "S355JR", 1, tuple(rm))

    def chaud_comme_froid():
        for n in MA.LIGNES:
            MA.LIGNES[n]["creux_chaud"] = MA.LIGNES[n]["creux_froid"]

    def froid_comme_lamine():
        for n in MA.LIGNES:
            MA.LIGNES[n]["creux_froid"] = MA.LIGNES[n]["lamine"]

    def s460_m():            # S460 lu au tableau 3.5 (thermomécanique) : la ligne n'est pas reprise → la N de 3.4 forcée à 3.11
        MA.LIGNES["S460"]["lamine"] = ("3.11", "S460NH")

    def alias_perdu():
        MA.ALIAS.pop("S355 (fu 490)")

    def n_comme_s355():
        MA.ALIAS["S355 N/NL"] = "S355"

    def palier_63():
        reh = list(MA.TABLEAUX["3.3"][1]["S355JR"][0])
        reh[3] = (80.0, True, 335.0)
        remplace("3.3", "S355JR", 0, tuple(reh))

    cas = [("option b : palier à 40 mm (tableau 3.1)", option_b), ("f_u S355 = 510 (tableau 3.1)", fu_510),
           ("f_u S275 = 430 (tableau 3.1)", fu_430), ("borne « < 3 » lue « ≤ 3 »", borne_3_incluse),
           ("creux finis à chaud lus au tableau des formés à froid", chaud_comme_froid),
           ("creux formés à froid lus comme des laminés", froid_comme_lamine), ("S460 laminé lu dans le mauvais tableau", s460_m),
           ("le nom « S355 (fu 490) » oublié", alias_perdu), ("« S355 N/NL » lu comme un S355 de l'EN 10025-2", n_comme_s355),
           ("S355 de 63 à 80 mm à 335 (palier 63-80 oublié)", palier_63)]
    res = []
    for nom, appliquer in cas:
        try:
            appliquer()
            try:
                rouges = [n for n, ok, _ in controles() if not ok]
            except Exception as e:  # noqa: BLE001
                rouges = [f"exception {type(e).__name__}: {e}"]
        finally:
            restaure()
        res.append((nom, rouges))
    return res


def main():
    print("=== 1, 2, 4, 5. Les valeurs, les cas du brief, les noms enregistrés, la référence ===")
    for n, ok, info in controles():
        chk(n, ok, info)
    print("\n=== 3. La forme ===")
    for x in forme():
        chk(*x)
    print("\n=== 6. Source unique ===")
    for x in source_unique():
        chk(*x)
    print("\n=== 7. Mutations ===")
    for nom, rouges in mutations():
        chk(f"mutation « {nom} » : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")
    chk("hors mutation, tous les contrôles de référence sont au vert", all(ok for _, ok, _ in controles()))
    print("\n=== 8. Pureté ===")
    code = ("import sys\nimport acier.materiaux\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] not in sys.stdlib_module_names\n"
            "             and m not in ('__main__', 'sitecustomize', '_distutils_hack')))")  # crochets de l'interpréteur
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    mods = out.stdout.strip()
    chk("la table se charge avec la seule bibliothèque standard (aucun module tiers chargé)",
        out.returncode == 0 and mods in ("['acier', 'acier.materiaux']",), (out.returncode, mods, out.stderr[-300:]))
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())

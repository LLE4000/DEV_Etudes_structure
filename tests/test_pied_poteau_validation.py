# -*- coding: utf-8 -*-
"""Validation croisée du pied de poteau (lot C) : le script
``tools/validation_pied_poteau.py`` — grille de ressorts et programmation
linéaire, hors moteur — rend VALIDÉ en mode rapide (12 configurations).

Il faut numpy et scipy (hors chemin de production) : sans eux, le test le
dit et passe (la MESURE de référence est le rapport daté de
docs/validation/, produit par le script complet).

Lancement : python3 tests/test_pied_poteau_validation.py
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tools"))

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


try:
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    DEPS = True
except ImportError:
    DEPS = False

print("=== Validation croisée (grille de ressorts + programme linéaire) ===")
if not DEPS:
    print("  SAUT  numpy / scipy absents : le script complet a produit docs/validation/pied_poteau_validation_*.md")
    chk("le rapport daté de la validation croisée existe",
        any(f.startswith("pied_poteau_validation_") for f in os.listdir(os.path.join("docs", "validation"))))
else:
    import validation_pied_poteau as V
    verdict = V.main(rapide=True)
    chk("mode rapide (12 configurations) : VALIDÉ — répartition, domaine, niveau 1 ≤ niveau 2", verdict)
    chk("figures produites (mode rapide, à côté de la mesure complète)",
        all(os.path.exists(os.path.join(V.FIG, f)) for f in
            ("pied_validation_A_repartition_rapide.png", "pied_validation_B_domaines_rapide.png",
             "pied_validation_ecarts_rapide.png")))
    for f in os.listdir(V.FIG):
        if f.endswith("_rapide.png"):
            os.remove(os.path.join(V.FIG, f))
    rap = os.path.join(V.DOSSIER, "pied_poteau_validation_rapide.md")
    if os.path.exists(rap):
        os.remove(rap)
chk("le moteur de production n'importe ni numpy ni scipy",
    not any(m in open(os.path.join("acier", "assemblages", "pied_poteau", "platine_beton", f), encoding="utf-8").read()
            for f in ("moteur.py", "equilibre.py", "geometrie.py", "composants.py") for m in ("import numpy", "import scipy")))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

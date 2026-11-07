#!/bin/sh
# Relance toute la lecture critique, dans l'ordre (≈ 10 min, dont ~6 pour les éléments finis au pas de 0,5 mm).
# À lancer depuis un dossier qui n'est PAS source/ (le classeur est une donnée, jamais exécuté).
D=$(dirname "$0")
for s in c01_classeur c02_conge c03_ef c03b_convergence c03c_mur c04_ancrage c05_cisaillement c06_ec3 c07_largeur \
         c08_generalisation c09_site; do
  echo "=== $s"; /usr/bin/python3.12 -I "$D/$s.py" || exit 1
done

#!/bin/sh
# Relance complète de la synthèse : calculs, figures (SVG puis PNG), assemblage du document, contrôle des chiffres.
# Les scripts sont lancés depuis /tmp (jamais depuis le dossier du classeur), en mode isolé, sans fichier compilé.
set -e
ICI="$(cd "$(dirname "$0")" && pwd)"
cd /tmp
for s in s01_classeur s02_controle s03_continus s04_retournee s05_longue s06_elu s07_maillage s08_catalogue s09_reprises \
         f_a_coupe f_b_diagrammes f_c_retournee f_d_longue f_e_maillage; do
  echo "== $s"
  /usr/bin/python3.12 -I -B -W ignore "$ICI/$s.py" > /dev/null
done
cd "$ICI" && node rendre_png.cjs ../figures/fig_a_coupe_A.svg ../figures/fig_b_diagrammes.svg ../figures/fig_c_retournee.svg \
  ../figures/fig_d_longue.svg ../figures/fig_e_maillage.svg
cd /tmp
/usr/bin/python3.12 -I -B "$ICI/assembler.py"
/usr/bin/python3.12 -I -B "$ICI/verifier_chiffres.py"

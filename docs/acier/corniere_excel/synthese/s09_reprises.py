# -*- coding: utf-8 -*-
"""S09 — Chiffres repris des autres analyses du dossier, relus dans LEURS sorties et recalculés ici quand ils sont
dérivés (aucune recopie à la main).

- support souple : scripts/c03c_mur.py → sorties/c03c_mur.json (analyse critique) ;
- convergence du second maillage : scripts/c03b_convergence.py → sorties/c03b_convergence.json ;
- seuil de Y sans cisaillement : scripts/c05_cisaillement.py → sorties/c05_cisaillement.json ;
- temps dans Pyodide : extensions/e06_pyodide.mjs → extensions/e06_sortie.json ;
- cornière PRS : extensions/e07_architecture.py → extensions/e07_sortie.txt.

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/s09_reprises.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s_commun as K  # noqa: E402

S = K.Sortie("s09_reprises")
h = K.historique(K.CAS)
R = K.RACINE

mur = json.load(open(os.path.join(R, "sorties", "c03c_mur.json"), encoding="utf-8"))
S("# S09 — chiffres repris des autres analyses")
S("")
S("## Support souple (analyse critique, c03c_mur) : contrainte nominale linéarisée de la section 3 face au classeur")
S("   (modèle de c03c : rondelle Ø30 RIGIDE appuyée de y = 70 à 100 mm, cheville δ = 0,1 mm, cisaillement en e/2)")
for c in mur:
    k = "rigide" if c["k_mur"] is None else f"{c['k_mur']:.0f} N/mm³"
    # statique du corps libre, cisaillement sur l'axe (x_V = e/2 dans ef_coupe) : R·(X − e/2) = T·(y_T − y_c)
    yT = c["yc"] + K.CAS["R"] * (K.CAS["X"] - K.CAS["e"] / 2) / (c["T_kN"] * 1e3)
    S(f"   mur {k:>12s} : pivot à {c['yc']:.2f} mm ; traction {c['T_kN']:.2f} kN ; résultante d'ancrage à y_T = {yT:.1f} mm "
      f"(axe 85) ; S3 nominale {c['S3_lin']:.2f} N/mm² ({100 * (c['S3_lin'] / h['sig'][2] - 1):+.1f} % sur le classeur {h['sig'][2]:.2f})")
S("   lecture : la résultante d'ancrage passe du bord BAS de la rondelle (mur rigide) à son bord HAUT (30 N/mm³) ; le « +3,8 % » "
  "vient de ce saut, propre à la rondelle rigide, pas de la seule montée du pivot (voir s07 : écrou ± 3 mm)")

conv = json.load(open(os.path.join(R, "sorties", "c03b_convergence.json"), encoding="utf-8"))
S("")
S("## Second maillage (analyse critique, c03b_convergence) : pic au congé selon le pas")
for c in conv:
    S(f"   pas {c['h']:.2f} mm ({c['noeuds']} nœuds) : pic de contrainte principale au congé {c['s1_conge']:.1f} N/mm² ; "
      f"traction {c['T_kN']:.2f} kN")

cis = json.load(open(os.path.join(R, "sorties", "c05_cisaillement.json"), encoding="utf-8"))
S("")
S(f"## Seuil de Y sans cisaillement (analyse critique, c05) : {cis['Y_limite']:.2f} mm")

e06 = json.load(open(os.path.join(R, "extensions", "e06_sortie.json"), encoding="utf-8"))
S("")
S(f"## Pyodide {e06.get('python')} + numpy {e06['numpy']} sous Node (extensions, e06)")
for c in e06["cas"]:
    S(f"   {c['noeuds']} nœuds : {c['t_total']:.2f} s ; traction {c['H_kN']:.3f} kN")
S(f"   chargement de Pyodide et numpy : {e06['t_chargement_pyodide_numpy']:.2f} s")

e07 = open(os.path.join(R, "extensions", "e07_sortie.txt"), encoding="utf-8").read()
S("")
S("## Géométrie partagée (extensions, e07)")
for l_ in e07.splitlines():
    if re.match(r"^(L120|PRS)", l_):
        S("   " + l_)
S.ecrire()

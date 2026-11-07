# -*- coding: utf-8 -*-
"""V07 — Point 7 de la relecture : tableau « raideur axiale d'un ancrage → traction H » (effet de levier).

Le tableau du §1.3 mélange-t-il deux maillages ? On refait chaque colonne au maillage GROS (n_t = 6, n_c = 8,
n_leg = 20, celui de s07) et au maillage FIN du document (n_t = 10, n_c = 16, n_leg = 40), mur raide (1e5 N/mm³),
écrou ± 3 mm. Plus la statique de référence (contact au talon, cisaillement sur l'axe) : H = R·(X − e/2)/Y.

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/v07_levier_maillage_fin.py
"""
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(ICI), "extensions"))
import ef_l  # noqa: E402

R, X, Y, e, r, L = 42760.0, 75.0, 85.0, 15.0, 13.0, 300.0
q = R / L
GEO = dict(B=120.0, H=120.0, e=e, r=r, Y=Y)
GROS = dict(n_t=6, n_c=8, n_leg=20, grad=4.0)
FIN = dict(n_t=10, n_c=16, n_leg=40, grad=4.0)
print(f"statique (contact au talon, V sur l'axe) : H = {R * (X - e / 2) / Y / 1e3:.2f} kN")
for kc in (30.0, 100.0, 200.0, 250.0, 300.0, 1000.0, 1e5, None):
    lig = []
    for nom, mail in (("gros", GROS), ("fin", FIN)):
        k_anc = 1e7 if kc is None else kc * 1e3 * 3 / L
        pb = ef_l.Probleme(GEO, "A", maillage=mail, solveur="bande")
        pb.resoudre(q, X, k_anc=k_anc, k_mur=1e5, b_rond=3.0)
        rr = pb.reactions
        tete = rr["mur_f"][rr["mur_y"] >= Y].sum() * L
        lig.append(f"{nom} : H = {rr['H'] * L / 1e3:6.2f} kN, tête {tete / 1e3:6.2f} kN")
    nomk = "infinie (k_anc = 1e7 N/mm par mm)" if kc is None else f"{kc:g} kN/mm"
    print(f"cheville {nomk:>34s} | " + " | ".join(lig), flush=True)

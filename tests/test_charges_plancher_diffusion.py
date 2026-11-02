# -*- coding: utf-8 -*-
"""Charges sur plancher — diffusion (plancher/charges/diffusion.py).

Chaque garantie rougit si on retire la formule qu'elle verrouille :
  1. Heft 240 : t_y = b0 + 2·h1 + h, plafonnée à 0,8·l ; b_v = t_y + 0,5·a
     (0,3·a en porte-à-faux), plafonnée ; b_m = t_y tant que k n'est pas
     transcrit, t_y + k·x·(1 − x/l) sinon (P-4).
  2. Lignes d'influence : P·L/4 à mi-portée, P·a·b/L sous la charge,
     réactions qui somment à P, w·L²/8 et w·L/2.
  3. Règle du levier : part proportionnelle au recouvrement, somme 1,
     troncature en rive renormalisée, largeur nulle → l'élément qui contient.
  4. Parts de l'annexe C fournies : voisins symétriques, somme 1, la part
     d'un voisin absent revient à l'élément chargé.
  5. Créneaux : val/w pour une boîte, somme dans le recouvrement, boîte
     tronquée en rive concentrée sur la largeur restante, liste vide → 0.

Lancement : python3 tests/test_charges_plancher_diffusion.py (depuis la racine).
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from plancher.charges import diffusion as DF  # noqa: E402
from plancher.charges.modele import Element  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))


def pres(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# 1. Heft 240
chk("t_y = b0 + 2h1 + h", pres(DF.t_y(0.3, 0.05, 0.2, 6.0), 0.6))
chk("t_y plafonnée à 0,8·l", pres(DF.t_y(3.0, 0.0, 0.2, 2.0), 1.6))
chk("b_v = t_y + 0,5·a", pres(DF.b_v(0.6, 2.0, 10.0), 1.6))
chk("b_v porte-à-faux 0,3·a", pres(DF.b_v(0.6, 2.0, 10.0, porte_a_faux=True), 1.2))
chk("b_v plafonnée", pres(DF.b_v(0.6, 2.0, 1.0), 1.0))
chk("b_m = t_y sans k (P-4)", pres(DF.b_m(0.6, 3.0, 6.0, None), 0.6))
chk("b_m = t_y + k·x(1 − x/l) avec k", pres(DF.b_m(0.6, 3.0, 6.0, 2.5), 0.6 + 2.5 * 3.0 * 0.5))
chk("b_m plafonnée", pres(DF.b_m(0.6, 3.0, 6.0, 2.5, b_max=2.0), 2.0))

# 2. lignes d'influence
chk("PL/4 à mi-portée", pres(DF.moment_ponctuel(10.0, 3.0, 3.0, 6.0), 15.0))
chk("P·a·b/L sous la charge", pres(DF.moment_ponctuel(10.0, 2.0, 2.0, 6.0), 10.0 * 2 * 4 / 6))
chk("moment linéaire à gauche", pres(DF.moment_ponctuel(10.0, 4.0, 1.0, 6.0), 10.0 * 1.0 * 2.0 / 6))
chk("moment linéaire à droite", pres(DF.moment_ponctuel(10.0, 2.0, 5.0, 6.0), 10.0 * 2.0 * 1.0 / 6))
chk("moment nul aux appuis", DF.moment_ponctuel(10.0, 2.0, 0.0, 6.0) == 0.0 and DF.moment_ponctuel(10.0, 2.0, 6.0, 6.0) == 0.0)
ra, rb = DF.reactions_ponctuelle(10.0, 2.0, 6.0)
chk("réactions 2/3 · 1/3", pres(ra, 10.0 * 4 / 6) and pres(rb, 10.0 * 2 / 6))
chk("w·L²/8", pres(DF.moment_uniforme(4.0, 3.0, 6.0), 4.0 * 36 / 8))
chk("w·L/2", pres(DF.tranchant_uniforme(4.0, 6.0), 12.0))

# 3. règle du levier
els = [Element(i, f"H{i + 1}", 1.2 * i, 1.2 * (i + 1)) for i in range(4)]
chk("levier : empreinte intérieure → 1", DF.parts_levier(0.6, 0.3, els) == {0: 1.0})
pl = DF.parts_levier(1.2, 0.6, els)
chk("levier : à cheval, 1/2 · 1/2", pres(pl[0], 0.5) and pres(pl[1], 0.5))
pl = DF.parts_levier(1.3, 0.6, els)
chk("levier : décalée, 1/3 · 2/3", pres(pl[0], 0.2 / 0.6) and pres(pl[1], 0.4 / 0.6))
chk("levier : somme 1", pres(sum(DF.parts_levier(2.35, 0.54, els).values()), 1.0))
pl = DF.parts_levier(0.1, 0.6, els)
chk("levier : rive tronquée renormalisée", pl == {0: 1.0})
chk("levier : largeur nulle → élément contenant", DF.parts_levier(2.5, 0.0, els) == {2: 1.0})
chk("levier : hors plancher → vide", DF.parts_levier(9.0, 0.3, els) == {})

# 4. parts annexe C
pv = DF.parts_voisins({0: 0.52, 1: 0.24}, 2, 5)
chk("annexe C : élément chargé et voisins", pres(pv[2], 0.52) and pres(pv[1], 0.24) and pres(pv[3], 0.24))
chk("annexe C : somme 1", pres(sum(pv.values()), 1.0))
pv = DF.parts_voisins({0: 0.52, 1: 0.24}, 0, 5)
chk("annexe C : rive, part du voisin absent à l'élément chargé", pres(pv[0], 0.76) and pres(pv[1], 0.24))
pv = DF.parts_voisins({0: 0.4, 1: 0.2, 2: 0.1}, 2, 5)
chk("annexe C : deux rangs de voisins", pres(pv[0], 0.1) and pres(pv[4], 0.1) and pres(pv[1], 0.2) and pres(pv[2], 0.4))
pv = DF.parts_voisins({0: 0.4, 1: 0.2}, 1, 3)
chk("annexe C : renormalisation quand les parts fournies ne somment pas à 1", pres(sum(pv.values()), 1.0) and pres(pv[1], 0.5))

# 5. créneaux
m, y = DF.max_creneaux([(2.0, 1.0, 10.0)], 0.0, 6.0)
chk("créneau seul : val/w", pres(m, 10.0) and 1.5 <= y <= 2.5)
m, y = DF.max_creneaux([(2.0, 1.0, 10.0), (2.4, 1.0, 10.0)], 0.0, 6.0)
chk("créneaux superposés : somme dans le recouvrement", pres(m, 20.0) and 1.9 <= y <= 2.5)
m, y = DF.max_creneaux([(0.1, 1.0, 10.0)], 0.0, 6.0)
chk("créneau tronqué en rive : concentré sur la largeur restante", pres(m, 10.0 / 0.6))
chk("créneaux : liste vide → 0", DF.max_creneaux([], 0.0, 6.0) == (0.0, 0.0))
m, _ = DF.max_creneaux([(1.0, 1.0, 10.0), (5.0, 1.0, 30.0)], 0.0, 6.0)
chk("créneaux disjoints : le plus fort", pres(m, 30.0))

print(f"RÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("  ÉCHEC :", nom, info)
sys.exit(1 if KO else 0)

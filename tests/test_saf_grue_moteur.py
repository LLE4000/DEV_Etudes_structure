# -*- coding: utf-8 -*-
"""Générateur SAF — fondation de grue : le MOTEUR DE CALCUL (saf/fondation_grue : geometrie, positions, plaque,
verification, proposition, projet). Cahier des charges du 14/11/2026, § 11 : les tests T1 à T13 ; lot 5 (15/11/2026, version
4) : T13, T16 et T18 à ses valeurs (le carré de diffusion à 45° depuis une plaque de 0,50 × 0,50, pour les deux types de plots).

Ce que la suite prouve :
  1. LES TESTS T1 À T13 — les valeurs du cahier des charges (maille 0,25 m, sol en compression seule, C2 = 0 sauf T4c).
  2. LA PARITÉ AVEC LE CALCUL DE RÉFÉRENCE — tests/fondation_grue_ref.py (annexe A, numpy + scipy), indépendant : le
     moteur du site résout sans scipy (par blocs, formule de Woodbury pour le sol en compression seule) et rend, cas
     par cas, la même contrainte, le même tassement, le même décollement, le même nombre d'itérations.
  3. LE PROJET — ses valeurs par défaut (celles du bureau ; réactions vides), sa relecture stricte, ses contrôles.
  4. MUTATIONS — chaque altération plausible d'une règle fait rougir un contrôle (arbre miroir, contrôles rapides).
  5. PURETÉ — le moteur se charge et calcule avec numpy seul : scipy, streamlit, pandas… refusés.

    /usr/bin/python3.12 tests/test_saf_grue_moteur.py
    /usr/bin/python3.12 tests/test_saf_grue_moteur.py --rapide [--racine dossier]   (les contrôles rapides seuls)
"""
import copy
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
ARGS = sys.argv[1:]
RACINE_MOTEUR = ARGS[ARGS.index("--racine") + 1] if "--racine" in ARGS else RACINE
RAPIDE = "--rapide" in ARGS
os.chdir(RACINE)
sys.path.insert(0, RACINE_MOTEUR)
sys.path.insert(1, ICI)

import numpy as np  # noqa: E402

from saf.fondation_grue import geometrie as G  # noqa: E402
from saf.fondation_grue import plaque as PL  # noqa: E402
from saf.fondation_grue import positions as PO  # noqa: E402
from saf.fondation_grue import projet as PR  # noqa: E402
from saf.fondation_grue import proposition as PP  # noqa: E402
from saf.fondation_grue import verification as V  # noqa: E402

OK, KO = [], []
JEUX = [(350, 440, 440, 350), (150, 150, 550, 550), (135, 100, 800, 135)]
E, K, H = 32800.0, 12.0, 0.5
L_T7 = [(0, 0), (12, 0), (12, 7), (7, 7), (7, 12), (0, 12)]


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info != "" and not cond else ""))


def rect(Lx, Ly):
    return [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]


def projet(jeux=JEUX, **k):
    """La dalle type du cahier des charges (8 × 8 × 0,50 m, C30/37, châssis saisi 6 × 6 m centré, appuis 0,5 × 0,5 m),
    modifiée par ``k`` : dalle, grue, options… (fusionnés dans le projet neuf)."""
    el = PR.nouveau()
    # (les tests T1 à T13 gardent la maille ÉCRITE dans chaque test, 0,25 m — cahier du lot 4, § 3.1 : le défaut est
    # désormais 0,20 m — et les bandes de 1,00 m de T10/T12)
    el["options"]["maille"], el["bandes"]["largeur"] = 0.25, 1.0
    el["reactions"] = [{"nom": "", "P": list(j)} for j in jeux] or [{"nom": "", "P": [None] * 4}]
    for cle, v in k.items():
        if isinstance(v, dict):
            el[cle].update(v)
        else:
            el[cle] = v
    return PR.lire(el)


def par_jeu(r):
    return [round(max(x["sigma_max"] for x in r["cas"] if x["jeu"] == j), 1) for j in (1, 2, 3)]


T7 = dict(dalle={"forme": "contour", "sommets": [list(p) for p in L_T7]},
          grue={"a": 4.5, "b": 4.5, "rect": {"a": 0.6, "b": 0.6, "angle": None}, "ex": -1.8, "ey": -1.8, "beta": 30.0})
PLOTS = {"type": "prefabrique", "a": 2.0, "b": 2.0, "h": 0.35}


def platine_45(xc, yc, sx, sy, a_trav, b_long):
    u, v = (sx / math.sqrt(2), sy / math.sqrt(2)), (-sy / math.sqrt(2), sx / math.sqrt(2))
    return [(xc + i * b_long / 2 * u[0] + j * a_trav / 2 * v[0], yc + i * b_long / 2 * u[1] + j * a_trav / 2 * v[1])
            for i, j in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


SIGNES = [(-1, 1), (1, 1), (1, -1), (-1, -1)]            # P1 haut-gauche … P4 bas-gauche


# ------------------------------------------------------------------ les contrôles RAPIDES (rejoués par les mutations)
def rapides():
    m = PR.modele(projet())
    r = V.calculer(m, K)
    chk("T4 dalle type : 16 positions ; σ max = 62,0 / 81,7 / 116,3 kPa (jeux 1 / 2 / 3) ; tassement max 9,70 mm ; pas de décollement ; "
        "distance mini plaque–bord 0,750 m", len(r["cas"]) == 16 and par_jeu(r) == [62.0, 81.7, 116.3] and round(r["w_max"], 2) == 9.70
        and r["decolle"] == 0 and round(m["controles"]["distance"], 3) == 0.750, (len(r["cas"]), par_jeu(r), r["w_max"], m["controles"]["distance"]))
    # (14/11/2026, tests/test_moteur_partage.py : CPython nommait « 3.f » le cas le plus défavorable, le navigateur « 3.g » —
    # sur la dalle type (carrée, châssis carré centré), les huit positions du jeu 3 sont ÉGALES par symétrie, et le bruit
    # de la virgule flottante les départageait autrement d'une machine à l'autre)
    j3 = [c["sigma_max"] for c in r["cas"] if c["jeu"] == 3]
    bruit = [62.0, 116.31 * (1 + 3e-13), 116.31, 50.0]
    chk("à ÉGALITÉ (les huit positions du jeu 3 de la dalle type : la même contrainte par symétrie, au bruit du calcul près), c'est la "
        "PREMIÈRE qui gouverne — « 3.a » —, sur toutes les machines ; premier_max départage à 1e-9 près, une vraie différence l'emporte "
        "toujours ; le premier nœud au maximum d'un champ uniforme est le nœud 0",
        r["cas_max"] == "3.a" and len(j3) == 8 and max(j3) - min(j3) <= 1e-9 * max(j3)
        and V.premier_max(bruit) == 1 and V.premier_max([116.31, 116.31 * (1 + 3e-13)]) == 0 and V.premier_max([1.0, 1.0 + 1e-6]) == 1
        and V.premier_argmax(np.full((3, 3), 12.5) + np.array([[0, 0, 0], [0, 0, 1e-13], [0, 0, 0]])) == 0,
        (r["cas_max"], min(j3), max(j3)))
    noms = [c["nom"] for c in m["cas"]]
    chk("§ 4 : les cas dans l'ordre du fichier — 1.a, 2.a, 3.a, 1.b, 2.b, 3.b, 1.c, 2.c, 3.c, 1.d, 2.d, 3.d, 3.e, 3.f, 3.g, 3.h ; LC2 est le "
        "premier ; jeu 1 → 4 positions, jeu 2 → 4, jeu 3 → 8",
        noms == ["1.a", "2.a", "3.a", "1.b", "2.b", "3.b", "1.c", "2.c", "3.c", "1.d", "2.d", "3.d", "3.e", "3.f", "3.g", "3.h"]
        and [c["lc"] for c in m["cas"]][:3] == ["LC2", "LC3", "LC4"] and m["cas"][-1]["lc"] == "LC17", noms)
    R = (1, 2, 3, 4)
    chk("§ 4 : rotation de k × 90° — le pied i reçoit R[(i + k) mod 4] ; miroir — R[(k − i) mod 4] ; carré : 8 positions d'un jeu quelconque ; "
        "rectangulaire : rotations 0 et 180°, miroirs selon les deux axes (4 positions)",
        PO.positions(R) == [(1, 2, 3, 4), (2, 3, 4, 1), (3, 4, 1, 2), (4, 1, 2, 3), (1, 4, 3, 2), (2, 1, 4, 3), (3, 2, 1, 4), (4, 3, 2, 1)]
        and PO.positions(R, False) == [(1, 2, 3, 4), (3, 4, 1, 2), (2, 1, 4, 3), (4, 3, 2, 1)], (PO.positions(R), PO.positions(R, False)))
    chk("§ 4 : la description d'un cas — « Réactions CAS 1.a (350/440/440/350) » ; le 3.d est (135/135/100/800) : l'ancien 3.b de l'Excel",
        m["cas"][0]["description"] == "Réactions CAS 1.a (350/440/440/350)" and m["cas"][11]["nom"] == "3.d"
        and m["cas"][11]["R"] == (135, 135, 100, 800), (m["cas"][0]["description"], m["cas"][11]))
    d = V.dalle(m, K)
    rond1 = [(1 + 0.25 * math.cos(i * math.pi / 24), 7 + 0.25 * math.sin(i * math.pi / 24)) for i in range(48)]
    brut = [float(d.charge(plaques=[(q, 1.0 / G.aire(q))])[0::3].sum()) for q in (rond1, G.rectangle(1.01, 7.02, 0.58, 0.40, 45.0))]
    chk("§ 5 : la charge d'une surface d'appui est NORMALISÉE — une platine ronde, une platine à 45° hors de l'alignement du maillage : la "
        "somme des forces verticales vaut exactement 1 (sans normaliser : 1,037 et 1,023, soit 3,7 % et 2,3 % d'écart)",
        all(abs(d.charge_unitaire(q)[0::3].sum() - 1.0) < 1e-12 for q in (rond1, G.rectangle(1.01, 7.02, 0.58, 0.40, 45.0)))
        and [round(x, 3) for x in brut] == [1.037, 1.023], brut)
    chk("T11 équilibre, dalle type : la réaction du sol = poids propre + réactions à moins de 0,01 % ; charges totales 1970, 2200, 2380 kN",
        r["ecart_equilibre"] < 1e-4 and sorted({round(x["charge_totale"], 1) for x in r["cas"]}) == [1970.0, 2200.0, 2380.0],
        (r["ecart_equilibre"], sorted({round(x["charge_totale"], 1) for x in r["cas"]})))
    r5 = V.calculer(PR.modele(projet(jeux=[])), K)
    chk("T5 poids propre seul (aucune réaction saisie) : 1 cas, σ = 25 × h = 12,5 kPa partout (12,48 à 12,51)",
        len(r5["cas"]) == 1 and round(r5["cas"][0]["sigma_min_lin"], 2) == 12.48 and round(r5["sigma_max"], 2) == 12.51,
        (len(r5["cas"]), r5["cas"][0]["sigma_min_lin"], r5["sigma_max"]))
    sm = V.sigma_moyen(m)
    chk("T5 σ moyen = (poids de la dalle + réactions du jeu le plus lourd) ÷ aire = 2380 / 64 = 37,2 kPa (jeu 1)",
        round(sm["sigma"], 1) == 37.2 and sm["jeu"] == 1 and sm["dalle"] + sm["reactions"] == 2380.0, sm)
    r8 = V.calculer(PR.modele(projet(jeux=[JEUX[2]])) | {"E": E * 1000}, K)
    chk("T8 dalle rigide (E × 1000) en compression seule, jeu 3 : σ max = 79,2 kPa, 7,2 % décollé (calcul rigide sans traction "
        "indépendant : 79,2 kPa, 7 %)", round(r8["sigma_max"], 1) == 79.2 and round(r8["decolle"], 1) == 7.2, (r8["sigma_max"], r8["decolle"]))
    b = [(x["nom"], tuple(round(v, 3) + 0.0 for v in x["debut"] + x["fin"])) for x in m["bandes"]]
    chk("T10 bandes, dalle type, largeur 1,00 m : BX1 (0 ; 7) → (8 ; 7), BX2 (0 ; 1) → (8 ; 1), BY1 (1 ; 0) → (1 ; 8), BY2 (7 ; 0) → (7 ; 8)",
        b == [("BX1", (0, 7, 8, 7)), ("BX2", (0, 1, 8, 1)), ("BY1", (1, 0, 1, 8)), ("BY2", (7, 0, 7, 8))]
        and all(x["largeur"] == 1.0 for x in m["bandes"]), b)
    m7 = PR.modele(projet(**T7))
    b7 = [(x["nom"], tuple(round(v, 3) + 0.0 for v in x["debut"] + x["fin"])) for x in m7["bandes"]]
    chk("T10 bandes, dalle en L (T7) : BX1 (0 ; 4,373) → (7 ; 8,415), BX2 (1,425 ; 0) → (12 ; 6,105), BY1 (4,027 ; 0) → (0 ; 6,975), "
        "BY2 (9,223 ; 0) → (2,295 ; 12) — limitées au contour, le morceau qui contient les deux plaques",
        b7 == [("BX1", (0, 4.373, 7, 8.415)), ("BX2", (1.425, 0, 12, 6.105)), ("BY1", (4.027, 0, 0, 6.975)), ("BY2", (9.223, 0, 2.295, 12))], b7)
    chk("T7 dalle en L, châssis 4,5 × 4,5 tourné de 30° : 16 positions, plaques dans la dalle, distance mini plaque–bord 0,717 m",
        len(m7["cas"]) == 16 and all(m7["controles"]["plaques_dedans"]) and round(m7["controles"]["distance"], 3) == 0.717,
        (len(m7["cas"]), m7["controles"]))
    U = [(0, 0), (12, 0), (12, 8), (8, 8), (8, 3), (4, 3), (4, 8), (0, 8)]
    bu = {x["nom"]: tuple(round(v, 3) + 0.0 for v in x["debut"] + x["fin"])
          for x in G.bandes_integration(U, [(9, 6.5), (11, 6.5), (11, 4.5), (9, 4.5)], 0.0, 1.0)}
    chk("§ 6 contour non convexe (une dalle en U, la grue sur la branche de droite) : la droite y = 6,5 traverse les deux branches — la "
        "bande BX1 garde le morceau qui contient les deux plaques, (8 ; 6,5) → (12 ; 6,5), pas celui de la branche de gauche",
        bu["BX1"] == (8, 6.5, 12, 6.5) and bu["BX2"] == (8, 4.5, 12, 4.5) and bu["BY1"] == (9, 0, 9, 8), bu)
    mt = PR.modele(projet(dalle=T7["dalle"], grue=dict(T7["grue"], plots={"type": "coule", "a": 1.5, "b": 1.0, "h": 0.35})))
    q0 = mt["plots"]["polys"][0]
    chk("§ 3.2 : un plot suit le châssis — tourné de 30° comme lui (son côté a fait 30° avec l'axe x), centré sur sa surface d'appui",
        round(math.degrees(math.atan2(q0[1][1] - q0[0][1], q0[1][0] - q0[0][0])), 6) == 30.0
        and round(math.hypot(q0[1][0] - q0[0][0], q0[1][1] - q0[0][1]), 9) == 1.5
        and all(abs(u - v) < 1e-9 for u, v in zip(G.centre(q0), G.centre(mt["plaques"][0]))), q0)
    cen = [(x + sx * 0.10 / math.sqrt(2), y + sy * 0.10 / math.sqrt(2)) for (x, y), (sx, sy) in zip(G.pieds(4, 4, 6, 6), SIGNES)]
    be = [(x["nom"], tuple(round(v, 3) + 0.0 for v in x["debut"] + x["fin"])) for x in G.bandes_integration(rect(8, 8), cen, 0.0, 1.0)]
    chk("T12 excentrement radial 0,10 m : les bandes passent par les centres des platines — BX1 et BX2 à y = 7,071 et 0,929 ; "
        "BY1 et BY2 à x = 0,929 et 7,071",
        be == [("BX1", (0, 7.071, 8, 7.071)), ("BX2", (0, 0.929, 8, 0.929)), ("BY1", (0.929, 0, 0.929, 8)), ("BY2", (7.071, 0, 7.071, 8))], be)
    pl45 = [platine_45(x, y, sx, sy, 0.58, 0.40) for (x, y), (sx, sy) in zip(G.pieds(4, 4, 6, 6), SIGNES)]
    rond = [[(x + 0.25 * math.cos(i * math.pi / 24), y + 0.25 * math.sin(i * math.pi / 24)) for i in range(48)] for x, y in G.pieds(4, 4, 6, 6)]
    chk("T12 distances mini : platines 0,58 × 0,40 à 45° → 0,654 m ; platines rondes Ø 0,50 en 48 côtés (aire 0,1958 m²) → 0,750 m",
        round(G.distance_plaques_bord(rect(8, 8), pl45)[0], 3) == 0.654 and round(G.distance_plaques_bord(rect(8, 8), rond)[0], 3) == 0.750
        and round(G.aire(rond[0]), 4) == 0.1958, (G.distance_plaques_bord(rect(8, 8), pl45), G.aire(rond[0])))
    chk("T13 contrôles : un plot de 2,00 m bord sur bord est dans la dalle (à 1 mm près), un plot de 2,20 m en sort ; deux plots qui se "
        "touchent ne se chevauchent pas, deux plots superposés de 10 cm oui",
        G.dans_dalle(rect(8, 8), G.rectangle(1, 1, 2.0, 2.0)) and not G.dans_dalle(rect(8, 8), G.rectangle(1, 1, 2.2, 2.2))
        and not G.se_chevauchent(G.rectangle(1, 1, 2, 2), G.rectangle(3, 1, 2, 2)) and G.se_chevauchent(G.rectangle(1, 1, 2, 2), G.rectangle(2.9, 1, 2, 2)))
    # (lot 5, § 6 : sous un plot — PRÉFABRIQUÉ OU COULÉ EN PLACE —, la réaction arrive sur le carré de diffusion à 45° depuis
    # une plaque de 0,50 × 0,50 — 1,20 × 1,20 m — : 66,1 / 85,8 / 120,6 kPa, 10,05 mm préfabriqués ; coulés en place (la raideur
    # du plot comptée) 64,7 / 84,0 / 114,9 kPa, 9,58 mm — au lot 4, la réaction d'un plot coulé restait sur la plaque :
    # 64,6 / 83,9 / 114,8 kPa)
    for typ, attendu, w in (("prefabrique", [66.1, 85.8, 120.6], 10.05), ("coule", [64.7, 84.0, 114.9], 9.58)):
        mp = PR.modele(projet(grue={"plots": dict(PLOTS, type=typ)}))
        rp = V.calculer(mp, K)
        chk(f"T13 dalle type, plots 2,00 × 2,00 × 0,35 m {'préfabriqués' if typ == 'prefabrique' else 'coulés en place'} : σ max = "
            f"{' / '.join(str(x).replace('.', ',') for x in attendu)} kPa ; tassement max {str(w).replace('.', ',')} mm ; charges totales "
            "2110, 2340, 2520 kN (le poids des plots s'ajoute) ; plots dans la dalle, séparés, qui couvrent les surfaces d'appui",
            par_jeu(rp) == attendu and round(rp["w_max"], 2) == w and sorted({round(x["charge_totale"], 1) for x in rp["cas"]}) == [2110.0, 2340.0, 2520.0]
            and mp["controles"]["plots_ok"] and rp["ecart_equilibre"] < 1e-4, (par_jeu(rp), rp["w_max"]))
    # T16 (lot 5, § 6) : le carré de diffusion à 45° depuis une plaque de 0,50 × 0,50 — côté 0,50 + 2 × hauteur du plot, sans
    # dépasser le plot, la base restant 0,50 quel que soit l'appui —, centré sur la surface d'appui
    def cote(q):
        return round(max(x for x, _ in q) - min(x for x, _ in q), 3), round(max(y for _, y in q) - min(y for _, y in q), 3)
    rond1 = [(1 + 0.25 * math.cos(i * math.pi / 24), 7 + 0.25 * math.sin(i * math.pi / 24)) for i in range(48)]
    d16 = [G.diffusion_45(G.rectangle(1, 7, 0.5, 0.5), 0.35, 2.0, 2.0), G.diffusion_45(G.rectangle(1, 7, 0.5, 0.5), 0.20, 2.0, 2.0),
           G.diffusion_45(G.rectangle(1, 7, 0.5, 0.5), 0.20, 0.8, 0.8), G.diffusion_45(platine_45(1, 7, -1, 1, 0.58, 0.40), 0.35, 2.0, 2.0),
           G.diffusion_45(rond1, 0.35, 2.0, 2.0)]
    centres16 = [tuple(round(v, 3) for v in G.centre_gravite(q)) for q in d16]
    chk("T16 carré de diffusion : plot 2,00 × 2,00 × 0,35 → 1,200 m ; 2,00 × 2,00 × 0,20 → 0,900 ; 0,80 × 0,80 × 0,20 → 0,800 (borné par "
        "le plot) ; 2,00 × 2,00 × 0,35 sous une platine 0,58 × 0,40 à 45° → 1,200 (la base reste 0,50 × 0,50), sous une platine ronde → 1,200 ; "
        "toujours centré sur la surface d'appui (1,000 ; 7,000), côtés parallèles au châssis",
        [cote(q) for q in d16] == [(1.2, 1.2), (0.9, 0.9), (0.8, 0.8), (1.2, 1.2), (1.2, 1.2)] and set(centres16) == {(1.0, 7.0)},
        ([cote(q) for q in d16], centres16))
    mc = PR.modele(projet(grue={"plots": dict(PLOTS, type="coule")}))
    chk("T16 les deux types : un plot coulé en place reçoit aussi la réaction sur son carré de diffusion de 1,20 m (le SAF et le calcul : "
        "verification.surfaces_chargees)", [round(G.aire(q), 6) for q in V.surfaces_chargees(mc)] == [1.44] * 4, V.surfaces_chargees(mc))
    mp = PR.modele(projet(grue={"plots": dict(PLOTS), "beta": 30.0}, dalle={"Lx": 10.0, "Ly": 10.0}))
    q0, c0 = mp["plots"]["diffusion"][0], G.centre_gravite(mp["plaques"][0])
    ang = math.degrees(math.atan2(q0[1][1] - q0[0][1], q0[1][0] - q0[0][0]))
    chk("T16 le modèle d'une dalle 10 × 10 sous un châssis tourné de 30°, plots préfabriqués : quatre carrés de 1,20 m, centrés sur les "
        "surfaces d'appui, tournés avec le châssis",
        len(mp["plots"]["diffusion"]) == 4 and abs(G.aire(q0) - 1.44) < 1e-9 and abs(ang - 30.0) < 1e-9
        and math.dist(G.centre_gravite(q0), c0) < 1e-9, (G.aire(q0), ang))
    mg = PR.modele(projet(grue={"plots": dict(PLOTS, a=2.2, b=2.2)}))
    chk("§ 3.2 : un plot de 2,20 m sous un pied à 1 m du bord sort de la dalle — contrôle rouge, « plots_ok » faux",
        not mg["controles"]["plots_ok"] and mg["controles"]["plots_dedans"] == [False] * 4)
    el = projet()
    # (lot 5, § 4 et § 12 : maille 0,20 m, bandes de 0,50 m, contraintes de sol en kg/cm², enrobage 5 cm, treillis CHOISIS Ø12 e 15
    # en bas et en haut, renforts Ø automatique, ancrage 50 Ø ; plus de torsion, de recouvrement ni de longueur maxi — retirés)
    chk("le projet neuf : dalle 8,00 × 8,00 × 0,50 m en C30/37 (E = 32 800 MPa), k = 2, 5, 10, 20, 30 dont 2, 5, 10 cochés « SCIA », châssis "
        "saisi 6,00 × 6,00 m centré à 0°, appuis 0,50 × 0,50 m, sans plot, distance mini 0,50 m, bandes de 0,50 m, limite 0,5 kg/cm², maille "
        "0,20 m, anneau 2 cm, C = k/10, unité kg/cm², enrobage 5 cm, treillis Ø12 e 15 en bas et en haut, renforts Ø auto, ancrage 50 Ø, "
        "efforts SCIA vides, aucune étape faite ; réactions et textes du projet VIDES",
        (lambda n: n["dalle"] == {"forme": "rectangle", "Lx": 8.0, "Ly": 8.0, "sommets": [], "h": 0.5, "beton": "C30/37"}
         and PR.module_E("C30/37") == 32800.0 and [(x["k"], x["scia"]) for x in n["sol"]["k"]] == [(2, True), (5, True), (10, True), (20, False), (30, False)]
         and n["grue"]["chassis"] == "saisie" and (n["grue"]["a"], n["grue"]["b"], n["grue"]["ex"], n["grue"]["ey"], n["grue"]["beta"], n["grue"]["dmin"]) == (6, 6, 0, 0, 0, 0.5)
         and n["grue"]["plots"]["type"] == "aucun" and n["bandes"]["largeur"] == 0.5 and n["limite"] == 0.5 and n["options"]["maille"] == 0.20
         and n["options"]["unite"] == "kgcm2" and not {"torsion", "recouvrement", "l_max", "gamma_s", "jeu", "arrondi"} & set(n["options"])
         and n["armatures"] == {"enrobage": 0.05, "inf": {"x": {"d": 12, "e": 150}, "y": {"d": 12, "e": 150}, "xy": False},
                                "sup": {"x": {"d": 12, "e": 150}, "y": {"d": 12, "e": 150}, "xy": False}, "renfort": "auto", "ancrage": 50.0}
         and all(v is None for k_, v in n["scia"].items() if k_ != "sigma") and n["scia"]["sigma"] == {} and n["etapes"] == {"saf": None, "note": None}
         and n["options"]["anneau"] == 0.02 and [n["options"][c] for c in ("c1x", "c1y", "c2x", "c2y")] == [0.1] * 4
         and n["reactions"] == [{"nom": "", "P": [None] * 4}] and set(n["projet"].values()) == {""})(PR.lire(PR.nouveau())))
    refus = []
    for cle, v in (("dalle", {"Lx": -1}), ("dalle", {"h": 0}), ("limite", 0), ("grue", {"chassis": "inconnu"}), ("bandes", {"largeur": 0.7}),
                   ("options", {"anneau": 0.05}), ("dalle", {"forme": "contour", "sommets": [[0, 0], [4, 4], [4, 0], [0, 4]]})):
        try:
            projet(**{cle: v})
            refus.append((cle, v))
        except ValueError:
            pass
    chk("la relecture refuse une valeur hors domaine, avec un message : longueur négative, épaisseur nulle, limite nulle, châssis inconnu, "
        "bande de 0,70 m, anneau de 5 cm, contour qui se recoupe", not refus, refus)
    chk("un jeu incomplet (une réaction non saisie) ne donne aucun cas ; les jeux gardent leur numéro de la liste : jeux 1 et 3 saisis → "
        "1.a, 3.a…", [c["nom"] for c in PR.modele(projet(jeux=[JEUX[0], (150, None, 550, 550), JEUX[2]]))["cas"]][:4] == ["1.a", "3.a", "1.b", "3.b"])
    chk("les classes de béton : E = 22·((f_ck + 8)/10)^0,3 GPa arrondi à 100 MPa — C20/25 30 000, C25/30 31 500, C30/37 32 800, C40/50 35 200, "
        "C50/60 37 300", [PR.module_E(c) for c in ("C20/25", "C25/30", "C30/37", "C40/50", "C50/60")] == [30000.0, 31500.0, 32800.0, 35200.0, 37300.0],
        [PR.module_E(c) for c in PR.BETONS])
    a = PR.agrandi(projet(), 0.35)
    ac = PR.agrandi(projet(**T7), 0.5)
    chk("« Proposer » agrandit un rectangle du même d de chaque côté (8 → 8,70 m, la grue reste centrée) et décale un contour vers "
        "l'extérieur (la dalle en L, 48 m de périmètre : + 25,0 m² pour 0,50 m — 24 m² de bande, cinq coins saillants, un rentrant)",
        a["dalle"]["Lx"] == 8.7 and a["dalle"]["Ly"] == 8.7 and PR.modele(a)["centre"] == (4.35, 4.35)
        and round(G.aire([tuple(p) for p in ac["dalle"]["sommets"]]) - G.aire(L_T7), 6) == 25.0,
        (a["dalle"], G.aire([tuple(p) for p in ac["dalle"]["sommets"]]) - G.aire(L_T7)))


# ------------------------------------------------------------------ les contrôles COMPLETS
def complets():
    import fondation_grue_ref as REF
    print("== 1. les tests T1 à T13 (suite)")
    d = PL.Dalle(rect(8, 8), H, K)
    s = d.resoudre([d.charge(q_uniforme=12.5)])[0]["sigma"]
    chk("T1 charge uniforme 12,5 kPa, dalle 8 × 8 × 0,50, k = 12 : σ = 12,5 kPa à ± 0,2 % (12,484 à 12,508)",
        round(s.min(), 3) == 12.484 and round(s.max(), 3) == 12.508, (s.min(), s.max()))
    d = PL.Dalle(rect(28, 28), H, K)
    F = np.zeros(d.ndof)
    c = (d.ny // 2) * (d.nx + 1) + d.nx // 2
    F[3 * c] = 1000.0
    s = d.resoudre([F], sans_traction=False)[0]["sigma"]
    hertz = 1000.0 / (8 * math.sqrt(K * 1000 * E * 1000 * H ** 3 / (12 * (1 - 0.04)))) * K * 1000
    chk("T2 P = 1000 kN au nœud central d'une dalle 28 × 28 × 0,50, sol linéaire : σ max = 22,98 kPa (Hertz P/8ℓ² : 22,95)",
        round(s[c], 2) == 22.98 and round(hertz, 2) == 22.95, (s[c], hertz))
    r = V.calculer(PR.modele(projet(jeux=[JEUX[2]])) | {"E": E * 1000}, K, sans_traction=False)
    chk("T3 dalle type, E × 1000, jeu 3, sol linéaire : σ max = 77,5 ; σ min = −15,8 kPa (dalle rigide à la main : 77,4 / −15,9)",
        round(r["cas"][0]["sigma_max"], 1) == 77.5 and round(r["cas"][0]["sigma_min_lin"], 1) == -15.8, r["cas"][0])
    r = V.calculer(PR.modele(projet(jeux=[JEUX[2]])), K, K / 10, K / 10)
    chk("T4c dalle type, jeu 3, avec C2x = C2y = k/10 : σ max = 114,9 kPa", round(r["sigma_max"], 1) == 114.9, r["sigma_max"])
    r = V.calculer(PR.modele(projet(jeux=[JEUX[2]], grue={"ex": 0.25, "ey": 0.25})), K)
    chk("T4b grue décalée de + 0,25 m en x et en y, jeu 3 : σ max = 134,9 kPa sur 8 positions (les 2 positions de l'ancien Excel : 116,0)",
        round(r["sigma_max"], 1) == 134.9 and len(r["cas"]) == 8, (r["sigma_max"], len(r["cas"])))
    m7 = PR.modele(projet(**T7))
    r7 = V.calculer(m7, K)
    chk("T7 dalle en L, 3 jeux : σ max = 62,7 kPa", round(r7["sigma_max"], 1) == 62.7, r7["sigma_max"])
    t = time.time()
    rk = V.par_k(PR.modele(projet()), (2, 5, 10, 20, 30), (0.0, 0.0))
    dt = time.time() - t
    chk("T9 dalle type, 3 jeux, k = 2 ; 5 ; 10 ; 20 ; 30 : σ max = 86,6 ; 97,0 ; 111,5 ; 132,2 ; 146,8 kPa — tassement max = 43,3 ; 19,4 ; "
        "11,1 ; 6,6 ; 4,9 mm — décollé = 4,0 ; 0,3 ; 0 ; 0 ; 0 %",
        [round(x["sigma_max"], 1) for x in rk] == [86.6, 97.0, 111.5, 132.2, 146.8] and [round(x["w_max"], 1) for x in rk] == [43.3, 19.4, 11.1, 6.6, 4.9]
        and [round(x["decolle"], 1) for x in rk] == [4.0, 0.3, 0.0, 0.0, 0.0], [(x["sigma_max"], x["w_max"], x["decolle"]) for x in rk])
    chk(f"… les cinq k en {dt:.2f} s (moins d'une seconde hors navigateur) ; la grue déplacée, la dalle n'est pas refaite", dt < 1.0, dt)
    base = PR.modele(projet())
    pl45 = [platine_45(x, y, sx, sy, 0.58, 0.40) for (x, y), (sx, sy) in zip(G.pieds(4, 4, 6, 6), SIGNES)]
    rond = [[(x + 0.25 * math.cos(i * math.pi / 24), y + 0.25 * math.sin(i * math.pi / 24)) for i in range(48)] for x, y in G.pieds(4, 4, 6, 6)]
    r45, rc = V.calculer(dict(base, plaques=pl45), K), V.calculer(dict(base, plaques=rond), K)
    chk("T12 dalle type, 3 jeux, surfaces d'appui de forme quelconque : platines 0,58 × 0,40 à 45° → σ max = 116,3 kPa ; platines rondes "
        "Ø 0,50 en 48 côtés → 116,3 kPa", round(r45["sigma_max"], 1) == 116.3 and round(rc["sigma_max"], 1) == 116.3, (r45["sigma_max"], rc["sigma_max"]))
    grande = dict(dalle={"Lx": 12.0, "Ly": 12.0, "h": 0.40})
    v = [round(V.calculer(PR.modele(projet(grue={"plots": pt}, **grande)), 30.0)["sigma_max"], 1)
         for pt in ({"type": "aucun"}, PLOTS, dict(PLOTS, type="coule"))]
    chk("T13 dalle 12 × 12 × 0,40, k = 30 (le maximum est sous le pied) : 53,1 kPa sans plot ; 52,7 préfabriqués ; 40,4 coulés en place "
        "(la raideur du plot comptée ; lot 5, § 6 : la réaction sur le carré de diffusion, 40,7 au lot 4)", v == [53.1, 52.7, 40.4], v)
    pl_type = [G.rectangle(x, y, 0.5, 0.5) for x, y in G.pieds(4, 4, 6, 6)]
    ch = [(q, R / 0.25) for q, R in zip(pl_type, JEUX[2])]
    s1, s2 = [dd.resoudre([dd.charge(q_uniforme=25 * 0.85, plaques=ch)])[0]["sigma"].max()
              for dd in (PL.Dalle(rect(8, 8), H, K, zones=[(rect(8, 8), 0.85)]), PL.Dalle(rect(8, 8), 0.85, K))]
    chk("T13 contrôle : une zone de 0,85 m sur toute la dalle = une dalle de 0,85 m (96,60 kPa)",
        round(s1, 2) == 96.60 and round(s2, 2) == 96.60, (s1, s2))
    equilibre = list(V.calculer(base, K)["cas"]) + list(r7["cas"])
    equilibre += V.calculer(PR.modele(projet(jeux=[JEUX[2]])) | {"E": E * 1000}, K)["cas"] + r45["cas"] + rc["cas"]
    for typ in ("prefabrique", "coule"):
        equilibre += V.calculer(PR.modele(projet(grue={"plots": dict(PLOTS, type=typ)})), K)["cas"]
    ecart = max(abs(x["reaction_sol"] - x["charge_totale"]) / x["charge_totale"] for x in equilibre)
    totaux = sorted({round(x["charge_totale"], 1) for x in equilibre})
    chk("T11 équilibre sur les 104 cas de T4, T7, T8, T12 et T13 : réaction du sol = poids propre + plots + réactions à moins de 0,01 % ; "
        "charges totales 1970 ; 2200 ; 2380 (dalle type), 2110 ; 2340 ; 2520 (avec plots) et 2657,5 ; 2887,5 ; 3067,5 kN (dalle T7)",
        len(equilibre) == 104 and ecart < 1e-4 and totaux == [1970.0, 2110.0, 2200.0, 2340.0, 2380.0, 2520.0, 2657.5, 2887.5, 3067.5],
        (len(equilibre), ecart, totaux))
    # T18 (lot 5, § 5.2) : la résultante du sol (Σ σ × aire des nœuds) contre grue + dalle + plots, cas 1.a, dalle type, k = 12
    t18 = []
    for pt in ({"type": "aucun"}, PLOTS, dict(PLOTS, type="coule")):
        m18 = PR.modele(projet(grue={"plots": pt}))
        c18 = next(x for x in V.calculer(m18, K)["cas"] if x["nom"] == "1.a")
        plots18 = sum(25 * m18["plots"]["hauteur"] * G.aire(q) for q in m18["plots"]["polys"]) if m18["plots"] else 0.0
        t18.append((round(c18["reaction_sol"], 1), round(sum(c18["R"]) + 25 * 0.5 * 64.0 + plots18, 1), abs(c18["reaction_sol"] / (sum(c18["R"]) + 800.0 + plots18) - 1)))
    chk("T18 équilibre, dalle type, jeu 1 (cas 1.a) : résultante du sol 2 380,0 kN sans plot ; 2 520,0 kN avec plots préfabriqués ou coulés "
        "(grue 1 580,0 + dalle 800,0 + plots 140,0) ; écart 0,00 %",
        [x[:2] for x in t18] == [(2380.0, 2380.0), (2520.0, 2520.0), (2520.0, 2520.0)] and max(x[2] for x in t18) < 5e-5, t18)
    t = time.time()
    t6 = [PP.pour_epaisseur(projet(jeux=[JEUX[2]]), h, ks=[K], rapport_c2=(0.0, 0.0)) for h in (0.4, 0.5, 0.6, 0.8, 1.0)]
    chk(f"T6 « Proposer », dalle type, jeu 3, k = 12, limite 0,5 kg/cm² : h = 0,40 → 10,50 m ; 0,50 → 10,70 ; 0,60 → 10,90 ; 0,80 → 11,30 ; "
        f"1,00 → 11,90 m ; σ max 47,9 à 48,8 kPa ≤ 49,03 ; béton 44,1 à 141,6 m³ ({time.time() - t:.1f} s)",
        [x["Lx"] for x in t6] == [10.5, 10.7, 10.9, 11.3, 11.9] and [round(x["sigma_max"], 1) for x in t6] == [47.9, 48.0, 48.5, 48.8, 48.8]
        and [round(x["volume"], 1) for x in t6] == [44.1, 57.2, 71.3, 102.2, 141.6] and PP.meilleure(t6) == 0, t6)
    chk("« Proposer » : les épaisseurs testées vont de 0,30 à 1,20 m par pas de 0,10 m (dix valeurs) ; une dalle qui tient déjà n'est pas "
        "agrandie (d = 0)", PR.epaisseurs(projet()) == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        and PP.pour_epaisseur(projet(jeux=[(50, 50, 50, 50)]), 0.5, ks=[K], rapport_c2=(0, 0))["d"] == 0.0)

    print("== 2. la parité avec le calcul de référence (tests/fondation_grue_ref.py : numpy + scipy)")
    def paire(nom, mien, ref):
        """Les cas du moteur et ceux de la référence, appariés par (jeu, réactions) : mêmes σ max, w max, décollement."""
        cle = lambda x: (x["jeu"], tuple(float(v) for v in x["R"]))                                       # noqa: E731
        par = {cle(x): x for x in ref}
        pire, iterations = 0.0, True
        for x in mien["cas"]:
            y = par[cle(x)]
            pire = max(pire, abs(x["sigma_max"] - y["sigma_max"]), abs(x["w_max"] - y["w_max"]), abs(x["decolle"] - y["decolle"]))
            iterations = iterations and x["iterations"] == y["iterations"]
        chk(f"parité · {nom} : {len(mien['cas'])} cas, plus grand écart {pire:.1e} (kPa, mm ou %), mêmes nombres d'itérations",
            len(mien["cas"]) == len(ref) and pire < 1e-5 and iterations, (len(mien["cas"]), len(ref), pire, iterations))
    paire("dalle type, k = 12", V.calculer(base, K), REF.verifier(rect(8, 8), H, K, (6, 6), (0.5, 0.5), JEUX))
    paire("dalle type, k = 2 (4 % décollé)", V.calculer(base, 2.0), REF.verifier(rect(8, 8), H, 2.0, (6, 6), (0.5, 0.5), JEUX))
    paire("dalle type, C2 = k/10", V.calculer(base, K, 1.2, 1.2), REF.verifier(rect(8, 8), H, K, (6, 6), (0.5, 0.5), JEUX, c2=1.2))
    paire("dalle en L, châssis tourné de 30°", r7, REF.verifier(L_T7, H, K, (4.5, 4.5), (0.6, 0.6), JEUX, centre=(4.2, 4.2), beta=30.0))
    paire("grue décalée", V.calculer(PR.modele(projet(grue={"ex": 0.25, "ey": 0.25})), K),
          REF.verifier(rect(8, 8), H, K, (6, 6), (0.5, 0.5), JEUX, centre=(4.25, 4.25)))
    paire("platines à 45°", r45, REF.verifier(rect(8, 8), H, K, (6, 6), (0.5, 0.5), JEUX, plaques=pl45))
    paire("platines rondes", rc, REF.verifier(rect(8, 8), H, K, (6, 6), (0.5, 0.5), JEUX, plaques=rond))
    for typ, mot in (("prefabrique", "préfabriqués"), ("coule", "coulés en place")):
        paire(f"plots {mot}", V.calculer(PR.modele(projet(grue={"plots": dict(PLOTS, type=typ)})), K),
              REF.verifier(rect(8, 8), H, K, (6, 6), (0.5, 0.5), JEUX, plots=(typ, 2.0, 2.0, 0.35)))
    paire("dalle 16 × 8 (blocs par colonnes), k = 5", V.calculer(PR.modele(projet(dalle={"Lx": 16.0})), 5.0),
          REF.verifier(rect(16, 8), H, 5.0, (6, 6), (0.5, 0.5), JEUX))
    paire("dalle rigide, 7 % décollé", V.calculer(PR.modele(projet()) | {"E": E * 1000}, K),
          REF.verifier(rect(8, 8), H, K, (6, 6), (0.5, 0.5), JEUX, E=E * 1000))
    # un décollement massif : au-delà du seuil, le moteur refactorise au lieu d'employer la formule de Woodbury
    lourd = [(0, 0, 3000, 0)]
    a_, b_ = V.calculer(PR.modele(projet(jeux=lourd)), 2.0), REF.verifier(rect(8, 8), H, 2.0, (6, 6), (0.5, 0.5), lourd)
    chk(f"parité · un pied très chargé sur sol mou ({a_['decolle']:.0f} % décollé : le moteur refactorise au-delà de "
        f"{100 * PL.SEUIL_WOODBURY:.0f} %) : même σ max ({a_['sigma_max']:.1f} kPa) et même décollement",
        a_["decolle"] > 100 * PL.SEUIL_WOODBURY and abs(a_["sigma_max"] - max(x["sigma_max"] for x in b_)) < 1e-5
        and abs(a_["decolle"] - max(x["decolle"] for x in b_)) < 1e-9, (a_["decolle"], a_["sigma_max"], max(x["sigma_max"] for x in b_)))


# ------------------------------------------------------------------ mutations (arbre miroir, contrôles rapides)
MUTATIONS = [
    ("verification.py", '''    pire = lignes[premier_max([x["sigma_max"] for x in lignes])]''', '''    pire = max(lignes, key=lambda x: x["sigma_max"])''',
     "le cas le plus défavorable départagé par le bruit du calcul (« 3.f » ici, « 3.g » ailleurs)"),
    ("positions.py", "candidats = [tuple(R[(i + k) % 4] for i in range(4)) for k in ks_rot]", "candidats = [tuple(R[(i - k) % 4] for i in range(4)) for k in ks_rot]",
     "la rotation dans l'autre sens (l'erreur de l'ancien Excel)"),
    ("positions.py", "ks_mir = (0, 1, 2, 3) if chassis_carre else (1, 3)", "ks_mir = () if chassis_carre else (1, 3)", "les miroirs oubliés"),
    ("positions.py", "        for j, pos in enumerate(par_jeu, 1):\n            if rang < len(pos):", "        for j, pos in enumerate(par_jeu, 1):\n            if rang == 0 or rang < len(pos) - 1:",
     "une position perdue"),
    ("plaque.py", "                nouveau = on0 & (w > 0)", "                nouveau = on0", "le sol retient la dalle (traction admise)"),
    ("plaque.py", "        return f / total", "        return f", "la charge d'une surface d'appui non normalisée"),
    ("plaque.py", "            self.De[dz] = E_MPa * 1000.0 * hz ** 3 / (12 * (1 - nu * nu))", "            pass", "le plot coulé en place sans sa surépaisseur"),
    ("verification.py", '    return pl["diffusion"] if pl else modele["plaques"]', '    return modele["plaques"]',
     "le plot qui ne diffuse pas la réaction (tout sur la platine)"),
    ("verification.py", '    return pl["diffusion"] if pl else modele["plaques"]',
     '    return pl["polys"] if pl else modele["plaques"]', "le plot qui répartit la réaction sur TOUT le plot (la règle d'avant le lot 4)"),
    ("verification.py", '    return pl["diffusion"] if pl else modele["plaques"]',
     '    return pl["diffusion"] if pl and pl["type"] == "prefabrique" else modele["plaques"]',
     "le plot coulé en place qui garde la réaction sur la platine (la règle du lot 4)"),
    ("geometrie.py", '    return rectangle(xc, yc, min(c, a_plot), min(c, b_plot), beta_deg)', '    return rectangle(xc, yc, c, c, beta_deg)',
     "le carré de diffusion qui dépasse le plot"),
    ("geometrie.py", '    c = base + 2.0 * h_plot', '    c = base + h_plot', "la diffusion à 45° d'un seul côté (côté + h au lieu de + 2h)"),
    ("geometrie.py", '    c = base + 2.0 * h_plot', '    c = math.sqrt(aire(surface)) + 2.0 * h_plot',
     "la diffusion depuis la surface d'appui au lieu d'une plaque de 0,50 × 0,50 (la règle du lot 4)"),
    ("verification.py", '            Fg = Fg + d.charge_unitaire(q) * (gamma * pl["hauteur"] * G.aire(q))', "            pass", "le poids des plots oublié"),
    ("verification.py", '    lourd = max(cas, key=lambda c: sum(c["R"])) if cas else None', '    lourd = cas[-1] if cas else None', "σ moyen sur un autre jeu que le plus lourd"),
    ("geometrie.py", "        f = 1 - tol / max(d, tol)", "        f = 1.0", "le bord sur bord refusé (plus de tolérance de 1 mm)"),
    ("geometrie.py", '        if t1 <= tm <= t2 and dans_polygone(mx, my, contour):', '        if dans_polygone(mx, my, contour):',
     "la bande d'un contour non convexe : le premier morceau au lieu de celui des plaques"),
    ("geometrie.py", 'NOMS_BANDES = (("BX1", (0, 1), "x"), ("BX2", (3, 2), "x"), ("BY1", (0, 3), "y"), ("BY2", (1, 2), "y"))',
     'NOMS_BANDES = (("BX1", (3, 2), "x"), ("BX2", (0, 1), "x"), ("BY1", (0, 3), "y"), ("BY2", (1, 2), "y"))', "BX1 et BX2 échangées"),
    ("projet.py", "    return round(22000.0 * ((fck + 8.0) / 10.0) ** 0.3 / 100.0) * 100.0", "    return round(22000.0 * ((fck + 8.0) / 10.0) ** 0.3)",
     "E non arrondi à 100 MPa"),
    ("projet.py", "    cx, cy = (xmin + xmax) / 2 + g[\"ex\"], (ymin + ymax) / 2 + g[\"ey\"]", "    cx, cy = xmin + g[\"ex\"], ymin + g[\"ey\"]",
     "la grue cotée depuis le coin au lieu du centre"),
    ("projet.py", '        polys = [G.rectangle(*G.centre_gravite(s), p["a"], p["b"], g["beta"]) for s in plaques]', '        polys = [G.rectangle(*G.centre_gravite(s), p["a"], p["b"], 0.0) for s in plaques]',
     "les plots qui ne suivent pas la rotation du châssis"),
]
TEMOIN = ("positions.py", 'LETTRES = "abcdefgh"', 'LETTRES = "abcdefgh"', "témoin : rien ne change")


def muter(fichier, ancien, nouveau):
    """Les contrôles rapides rejoués sur une copie du moteur où ``ancien`` est remplacé : (texte trouvé, code de sortie)."""
    with tempfile.TemporaryDirectory() as tmp:
        for paquet in ("saf", "grues"):
            shutil.copytree(os.path.join(RACINE, paquet), os.path.join(tmp, paquet), ignore=shutil.ignore_patterns("__pycache__"))
        # (le moteur lit les options par défaut du moteur poutre — beton/poutre, beton/position — lot 4 : lié, jamais muté)
        os.symlink(os.path.join(RACINE, "beton"), os.path.join(tmp, "beton"))
        os.symlink(os.path.join(RACINE, "beton_classes.json"), os.path.join(tmp, "beton_classes.json"))
        cible = os.path.join(tmp, "saf", "fondation_grue", fichier)
        src = open(cible, encoding="utf-8").read()
        if src.count(ancien) != 1:
            return False, None
        open(cible, "w", encoding="utf-8").write(src.replace(ancien, nouveau))
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--rapide", "--racine", tmp], capture_output=True, text=True, cwd=RACINE)
        return True, p.returncode


def main():
    print("== 1. les tests T1 à T13, le projet (contrôles rapides)")
    rapides()
    if RAPIDE:
        print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
        return 1 if KO else 0
    complets()
    print("== 4. mutations")
    trouve, code = muter(*TEMOIN[:3])
    chk("témoin : le moteur recopié sans rien changer passe les contrôles rapides", trouve and code == 0, code)
    for fichier, ancien, nouveau, nom in MUTATIONS:
        trouve, code = muter(fichier, ancien, nouveau)
        chk(f"mutation « {nom} » ({fichier}) : un contrôle rougit", trouve and code == 1,
            "texte absent ou répété" if not trouve else f"code {code} : AUCUN contrôle ne rougit")
    print("== 5. pureté")
    code = ("import sys\n"
            "class Refus:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('scipy', 'streamlit', 'pandas', 'matplotlib', 'reportlab', 'PIL', 'fitz', 'openpyxl'):\n"
            "            raise ImportError('refusé : ' + name)\n"
            "sys.meta_path.insert(0, Refus())\n"
            f"sys.path.insert(0, {RACINE!r})\n"
            "from saf.fondation_grue import projet as PR, verification as V\n"
            "el = PR.nouveau(); el['options']['maille'] = 0.25; el['reactions'] = [{'nom': '', 'P': [135, 100, 800, 135]}]\n"
            "print(round(V.calculer(PR.modele(PR.lire(el)), 12.0)['sigma_max'], 1))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    chk("le moteur se charge et calcule (116,3 kPa) avec numpy seul : scipy, streamlit, pandas, matplotlib, reportlab, openpyxl refusés",
        p.returncode == 0 and p.stdout.strip() == "116.3", (p.stdout.strip(), p.stderr[-300:]))
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom[:160], "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())

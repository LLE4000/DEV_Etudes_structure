# -*- coding: utf-8 -*-
"""Générateur SAF — fondation de grue : le PRINCIPE D'ARMATURES (saf/fondation_grue/armatures.py) et les EFFORTS SCIA recopiés
(saf/fondation_grue/scia.py). Cahier des charges du lot 5, version 4 (15/11/2026), § 7, § 8, § 13 et annexe A : les tests T19,
A1 à A4 (sous-lot 5b) et E1 à E4 côté moteur (sous-lot 5c ; l'écran : tests/test_api_web_saf.py).

Ce que la suite prouve :
  1. T19, A1 — avec mRd imposé à 97,3 kNm/m et l'ancrage de 0,80 m, les renforts des bandes et les zones rouges sont, à
     l'écriture près, la sortie de l'annexe A (8 cas, bandes de 0,50 et de 1,00 m, plots, SCIA × 1,10) ; le module s'en sert
     tel quel (son ancrage : 50 Ø).
  2. LE MOTEUR DALLES — mRd = A_s du treillis ÷ A_s,req(1 kNm/m) (Ø12 e 15 : 97,3 en x, 95,0 en y, la couche y sur la couche
     x) ; A3 : le nombre de barres de chaque renfort, ⌈(A_s(m_dim) − A_s du treillis) × largeur ÷ aire d'une barre⌉ avec
     A_s() du moteur ; la carte de l'écran montre les mêmes barres ; A4 : la hauteur de béton et l'effort tranchant, les deux
     seuls cas (avec le sol) où le module dit d'épaisser ; le Ø automatique (le plus léger), imposé ; châssis tourné.
  3. SCIA (E1 à E4) — les valeurs grises (T14), la symétrie « x = y », un champ vide reste vide (r = 1), r = 1,10 → l'échelle
     du principe, l'écart hors de 0,85 à 1,15.
  4. LA PARITÉ AVEC LE CALCUL DE RÉFÉRENCE — tests/fondation_grue_ref.py (``principe_armatures``, scipy.ndimage) sur des
     dalles qui ne sont pas celles du cahier (contour en L, châssis tourné de 30°, grue décalée, mRd de 60 à 140).
  5. A2 — changer un treillis ou la largeur des bandes refait le principe, jamais le calcul du sol.
  6. PURETÉ (numpy seul) et MUTATIONS (arbre miroir, contrôles rapides).

    /usr/bin/python3.12 tests/test_saf_grue_armatures.py
    /usr/bin/python3.12 tests/test_saf_grue_armatures.py --rapide [--racine dossier]   (les contrôles rapides seuls)
"""
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
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, RACINE_MOTEUR)
sys.path.insert(1, ICI)

import numpy as np  # noqa: E402

from saf.fondation_grue import armatures as AR  # noqa: E402
from saf.fondation_grue import efforts as EF  # noqa: E402
from saf.fondation_grue import projet as PR  # noqa: E402
from saf.fondation_grue import scia as SC  # noqa: E402
from saf.fondation_grue import verification as V  # noqa: E402

OK, KO = [], []
JEUX = [(350, 440, 440, 350), (150, 150, 550, 550), (135, 100, 800, 135)]
KS = (2.0, 5.0, 10.0)
MRD_ESSAI = {c: 97.3 for c in AR.CLES}          # T19 : le point d'essai du cahier
PREF = {"type": "prefabrique", "a": 2.0, "b": 2.0, "h": 0.35}
COULE = {"type": "coule", "a": 2.0, "b": 2.0, "h": 0.35}
L_T7 = [(0, 0), (12, 0), (12, 7), (7, 7), (7, 12), (0, 12)]

T19_ATTENDU = """
T19 dalle type 8 x 8 x 0,50, bandes 0,50 :
    inf_x : 4 renfort(s) de bande ; 1er BX1 x 0.00 a 2.05, y 6.75 a 7.25, m_dim 143.2 | 4 zone(s) rouge(s), 0.64 m2, max 117.4 ; 1re x 0.00 a 2.10, y 0.30 a 0.70
    inf_y : 4 renfort(s) de bande ; 1er BY1 x 0.75 a 1.25, y 0.00 a 2.05, m_dim 143.2 | 4 zone(s) rouge(s), 0.64 m2, max 117.4 ; 1re x 0.30 a 0.70, y 0.00 a 2.10
    sup_x : 2 renfort(s) de bande ; 1er BX1 x 1.90 a 6.10, y 6.75 a 7.25, m_dim 126.7 | 4 zone(s) rouge(s), 8.96 m2, max 140.8 ; 1re x 1.70 a 6.30, y 0.00 a 0.70
    sup_y : 2 renfort(s) de bande ; 1er BY1 x 0.75 a 1.25, y 1.90 a 6.10, m_dim 126.7 | 4 zone(s) rouge(s), 8.96 m2, max 140.8 ; 1re x 0.00 a 0.70, y 1.70 a 6.30
T19 dalle type, bandes 1,00 :
    inf_x : 4 renfort(s) de bande ; 1er BX1 x 0.00 a 2.00, y 6.50 a 7.50, m_dim 127.1 | 4 zone(s) rouge(s), 0.16 m2, max 100.8 ; 1re x 0.10 a 1.90, y 0.30 a 0.50
    inf_y : 4 renfort(s) de bande ; 1er BY1 x 0.50 a 1.50, y 0.00 a 2.00, m_dim 127.1 | 4 zone(s) rouge(s), 0.16 m2, max 100.8 ; 1re x 0.30 a 0.50, y 0.10 a 1.90
    sup_x : 2 renfort(s) de bande ; 1er BX1 x 1.95 a 6.05, y 6.50 a 7.50, m_dim 126.5 | 4 zone(s) rouge(s), 6.88 m2, max 140.8 ; 1re x 1.70 a 6.30, y 0.00 a 0.50
    sup_y : 2 renfort(s) de bande ; 1er BY1 x 0.50 a 1.50, y 1.95 a 6.05, m_dim 126.5 | 4 zone(s) rouge(s), 6.88 m2, max 140.8 ; 1re x 0.00 a 0.50, y 1.70 a 6.30
T19 dalle type, plots coules en place, bandes 0,50 :
    inf_x : bandes : treillis suffisant | 4 zone(s) rouge(s), 1.28 m2, max 199.6 ; 1re x 0.00 a 2.90, y 0.00 a 0.30
    inf_y : bandes : treillis suffisant | 4 zone(s) rouge(s), 1.28 m2, max 199.6 ; 1re x 0.00 a 0.30, y 0.00 a 2.90
    sup_x : 2 renfort(s) de bande ; 1er BX1 x 1.60 a 6.40, y 6.75 a 7.25, m_dim 129.5 | 4 zone(s) rouge(s), 13.76 m2, max 169.0 ; 1re x 0.90 a 7.10, y 0.00 a 0.70
    sup_y : 2 renfort(s) de bande ; 1er BY1 x 0.75 a 1.25, y 1.60 a 6.40, m_dim 129.5 | 4 zone(s) rouge(s), 13.76 m2, max 169.0 ; 1re x 0.00 a 0.70, y 0.90 a 7.10
T19 dalle 10,70 x 10,70 x 0,50, bandes 0,50 :
    inf_x : 4 renfort(s) de bande ; 1er BX1 x 1.10 a 3.60, y 8.10 a 8.60, m_dim 186.7 | 8 zone(s) rouge(s), 4.08 m2, max 167.8 ; 1re x 1.12 a 3.53, y 1.31 a 2.12
    inf_y : 4 renfort(s) de bande ; 1er BY1 x 2.10 a 2.60, y 1.10 a 3.60, m_dim 186.7 | 8 zone(s) rouge(s), 4.08 m2, max 167.8 ; 1re x 1.31 a 2.12, y 1.12 a 3.53
    sup_x : bandes : treillis suffisant | hors bandes : treillis suffisant
    sup_y : bandes : treillis suffisant | hors bandes : treillis suffisant
T19 dalle 10,70, bandes 1,00 :
    inf_x : 4 renfort(s) de bande ; 1er BX1 x 1.10 a 3.60, y 7.85 a 8.85, m_dim 170.1 | 8 zone(s) rouge(s), 2.12 m2, max 134.1 ; 1re x 1.12 a 3.53, y 1.31 a 1.92
    inf_y : 4 renfort(s) de bande ; 1er BY1 x 1.85 a 2.85, y 1.10 a 3.60, m_dim 170.1 | 8 zone(s) rouge(s), 2.12 m2, max 134.1 ; 1re x 1.31 a 1.92, y 1.12 a 3.53
    sup_x : bandes : treillis suffisant | hors bandes : treillis suffisant
    sup_y : bandes : treillis suffisant | hors bandes : treillis suffisant
T19 dalle 10,70, plots prefabriques, bandes 0,50 :
    inf_x : 4 renfort(s) de bande ; 1er BX1 x 1.05 a 3.65, y 8.10 a 8.60, m_dim 128.2 | 8 zone(s) rouge(s), 4.08 m2, max 125.6 ; 1re x 0.92 a 3.73, y 1.51 a 2.12
    inf_y : 4 renfort(s) de bande ; 1er BY1 x 2.10 a 2.60, y 1.05 a 3.65, m_dim 128.2 | 8 zone(s) rouge(s), 4.08 m2, max 125.6 ; 1re x 1.51 a 2.12, y 0.92 a 3.73
    sup_x : bandes : treillis suffisant | hors bandes : treillis suffisant
    sup_y : bandes : treillis suffisant | hors bandes : treillis suffisant
T19 dalle 10,70, plots coules en place, bandes 0,50 :
    inf_x : bandes : treillis suffisant | 8 zone(s) rouge(s), 0.33 m2, max 110.8 ; 1re x 2.53 a 4.33, y 1.31 a 1.51
    inf_y : bandes : treillis suffisant | 8 zone(s) rouge(s), 0.33 m2, max 110.8 ; 1re x 3.33 a 3.53, y 0.51 a 2.31
    sup_x : bandes : treillis suffisant | hors bandes : treillis suffisant
    sup_y : bandes : treillis suffisant | hors bandes : treillis suffisant
T19 dalle 10,70, bandes 0,50, SCIA = module x 1,10 en bas :
    inf_x : 4 renfort(s) de bande ; 1er BX1 x 1.05 a 3.65, y 8.10 a 8.60, m_dim 205.4 | 8 zone(s) rouge(s), 5.71 m2, max 184.5 ; 1re x 0.92 a 3.73, y 1.31 a 2.12
    inf_y : 4 renfort(s) de bande ; 1er BY1 x 2.10 a 2.60, y 1.05 a 3.65, m_dim 205.4 | 8 zone(s) rouge(s), 5.71 m2, max 184.5 ; 1re x 1.31 a 2.12, y 0.92 a 3.73
    sup_x : bandes : treillis suffisant | hors bandes : treillis suffisant
    sup_y : bandes : treillis suffisant | hors bandes : treillis suffisant
"""


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info != "" and not cond else ""))


def projet(L=8.0, h=0.5, largeur=0.5, plots=None, **k):
    """La dalle type du cahier (L × L × h, châssis 6 × 6 m centré, appuis 0,5 × 0,5 m, maille 0,20 m, 3 jeux), modifiée par
    ``k`` (dalle, grue, armatures… fusionnés)."""
    el = PR.nouveau()
    el["reactions"] = [{"nom": "", "P": list(j)} for j in JEUX]
    el["dalle"]["Lx"] = el["dalle"]["Ly"] = L
    el["dalle"]["h"] = h
    el["bandes"]["largeur"] = largeur
    if plots:
        el["grue"]["plots"] = dict(plots)
    for cle, v in k.items():
        el[cle].update(v)
    return PR.lire(el)


_EF = {}


def efforts(el, d_v=None):
    """(modèle, enveloppe des efforts) — d_v = h − 0,06 m (0,44 m pour h = 0,50 : celui de T14 et T19)."""
    cle = repr(el)
    if cle not in _EF:
        m = PR.modele(el)
        _EF[cle] = (m, EF.enveloppe(m, V.par_k(m, list(KS), (0.1, 0.1)), d_v if d_v is not None else el["dalle"]["h"] - 0.06))
    return _EF[cle]


def f2(v):
    return f"{round(v, 6) + 0.0:.2f}"


def lignes_t19(nom, pa):
    """L'écriture de l'annexe A (T19) d'un principe."""
    out = [f"T19 {nom} :"]
    for cle in AR.CLES:
        rb = [r for r in pa["bandes"] if r["cle"] == cle]
        if rb:
            r = rb[0]
            tb = (f"{len(rb)} renfort(s) de bande ; 1er {r['bande']} x {f2(r['x0'])} a {f2(r['x1'])}, "
                  f"y {f2(r['y0'])} a {f2(r['y1'])}, m_dim {r['m_dim']:.1f}")
        else:
            tb = "bandes : treillis suffisant"
        rl = pa["locaux"][cle]
        if rl:
            r = rl[0]
            tl = (f"{len(rl)} zone(s) rouge(s), {sum(x['aire'] for x in rl):.2f} m2, max {max(x['m_dim'] for x in rl):.1f} ; "
                  f"1re x {f2(r['x0'])} a {f2(r['x1'])}, y {f2(r['y0'])} a {f2(r['y1'])}")
        else:
            tl = "hors bandes : treillis suffisant"
        out.append(f"    {cle} : {tb} | {tl}")
    return out


CAS_T19 = [("dalle type 8 x 8 x 0,50, bandes 0,50", dict(), None),
           ("dalle type, bandes 1,00", dict(largeur=1.0), None),
           ("dalle type, plots coules en place, bandes 0,50", dict(plots=COULE), None),
           ("dalle 10,70 x 10,70 x 0,50, bandes 0,50", dict(L=10.7), None),
           ("dalle 10,70, bandes 1,00", dict(L=10.7, largeur=1.0), None),
           ("dalle 10,70, plots prefabriques, bandes 0,50", dict(L=10.7, plots=PREF), None),
           ("dalle 10,70, plots coules en place, bandes 0,50", dict(L=10.7, plots=COULE), None),
           ("dalle 10,70, bandes 0,50, SCIA = module x 1,10 en bas", dict(L=10.7), {"inf_x": 1.1, "inf_y": 1.1})]


def _geo(pa_bandes, pa_locaux):
    """La géométrie d'un principe, arrondie au µm (pour comparer deux principes)."""
    b = [(r["bande"], r["cle"], round(r["x0"], 6), round(r["x1"], 6), round(r["y0"], 6), round(r["y1"], 6), round(r["m_dim"], 6)) for r in pa_bandes]
    loc = {c: [(round(r["x0"], 6), round(r["x1"], 6), round(r["y0"], 6), round(r["y1"], 6), round(r["aire"], 6), round(r["m_dim"], 6))
               for r in lst] for c, lst in pa_locaux.items()}
    return b, loc


# ------------------------------------------------------------------ les contrôles RAPIDES (rejoués par les mutations)
def rapides():
    print("== 1. T19, A1 : le principe de l'annexe A (mRd = 97,3 kNm/m, ancrage 0,80 m)")
    attendu = T19_ATTENDU.strip("\n").split("\n")
    blocs = [attendu[i * 5:(i + 1) * 5] for i in range(8)]
    for (nom, k, ech), bloc in zip(CAS_T19, blocs):
        m, ef = efforts(projet(**k))
        obtenu = lignes_t19(nom, AR.principe(ef, m["contour"], MRD_ESSAI, 0.80, ech))
        chk(f"T19 {nom} : renforts des bandes et zones rouges de l'annexe A", obtenu == bloc,
            [(a, b) for a, b in zip(obtenu, bloc) if a != b][:2])
    # A1 : le module se sert du principe tel quel (son mRd, son ancrage 50 Ø, son Ø)
    el = projet()
    m, ef = efforts(el)
    fer = AR.calculer(el, m, ef)
    res = fer["res"]
    pa = AR.principe(ef, m["contour"], res["mRd"], 50 * 12 / 1000, None, fer["mRd_bandes"])
    rb = [r for r in fer["renforts"] if r["type"] == "bande"]
    rl = {c: [r for r in fer["renforts"] if r["type"] == "local" and r["cle"] == c] for c in AR.CLES}
    chk("A1 le principe du module (Ø12 partout, ancrage 50 Ø = 0,60 m, mRd du moteur Dalles) est celui de l'annexe : mêmes renforts "
        "de bande, mêmes renforts locaux, dans l'ordre des repères (le bas puis le haut, les bandes puis les locaux, x puis y)",
        set(fer["diametres"].values()) == {12} and _geo(rb, rl) == _geo(sorted(pa["bandes"], key=lambda r: (r["cle"][:3] != "inf", r["cle"])), pa["locaux"])
        and [r["rep"] for r in fer["renforts"]] == [f"R{i}" for i in range(1, len(fer["renforts"]) + 1)]
        and all(r["ancrage"] == 0.6 for r in fer["renforts"]), (fer["diametres"], len(rb)))
    print("== 2. le moteur Dalles : mRd, A3 (les barres), A4 (la hauteur), le Ø, châssis tourné")
    s1 = AR.section(el, {"inf_x": 1.0})["grandeurs"]
    s50 = AR.section(el, {"inf_x": 50.0})["grandeurs"]
    s_r = AR.section(el, {c: res["mRd"][c] for c in AR.CLES})["grandeurs"]
    chk("mRd du treillis Ø12 e 15 (enrobage 5 cm, h = 50 cm) par le moteur Dalles : 97,3 kNm/m en x (d = 43,0 cm), 95,0 en y (d = 42,0 cm : "
        "la couche y sur la couche x), les deux faces ; A_s = 754 mm²/m ; A_s,req est linéaire en m (50 kNm/m : 50 × celui de 1 kNm/m), et à "
        "m = mRd le moteur demande exactement le treillis",
        f2(res["mRd"]["inf_x"]) == "97.26" and f2(res["mRd"]["inf_y"]) == "95.00" and res["mRd"]["sup_x"] == res["mRd"]["inf_x"]
        and (res["d"]["inf_x"], res["d"]["inf_y"]) == (43.0, 42.0) and round(res["As"]["inf_x"]) == 754
        and abs(s50["As_req_inf_x"] - 50 * s1["As_req_inf_x"]) < 1e-9 * s50["As_req_inf_x"]
        and all(abs(s_r[f"As_req_{c}"] - res["As"][c]) < 1e-6 for c in AR.CLES), (res["mRd"], res["d"]))
    # A3 : la dalle 10,70 sans plot, Ø12 e 15
    e7 = projet(L=10.7)
    m7, ef7 = efforts(e7)
    fer7 = AR.calculer(e7, m7, ef7)
    faux = []
    for r in fer7["renforts"]:
        g = AR.section(e7, {r["cle_section"]: r["m_dim"]})["grandeurs"]
        besoin = max(g[f"As_req_{r['cle_section']}"], g[f"As_min_{r['cle_section']}"])
        # (la largeur relue sur le RECTANGLE du renfort, pas sur la valeur du module — revue du 15/11/2026 : une largeur
        #  prise pour la longueur passait)
        larg = ef7["largeur"] if r["type"] == "bande" else ((r["y1"] - r["y0"]) if r["sens"] == "x" else (r["x1"] - r["x0"]))
        n = math.ceil((besoin - 1000 * math.pi * 12 ** 2 / 4 / 150) * larg / (math.pi * r["d"] ** 2 / 4) - 1e-9)
        if n != r["n"] or abs(r["besoin"] - besoin) > 1e-9 or abs(r["largeur"] - larg) > 1e-9:
            faux.append((r["rep"], r["n"], n))
    from api_web import saf_armatures as SAW
    import re as RE
    sc7 = SC.rapports(e7, ef7, None)
    carte = SAW.carte(e7, fer7, ef7, sc7).replace(" ", " ")
    # les barres de la carte, renfort par renfort : « R1–R4 : 4 Ø12 », « R1, R3 : 4 Ø12 », « R5 3 Ø12, R6 4 Ø12 »
    lues = {}
    for grp, n_, d_ in RE.findall(r"(R\d+(?:–R\d+)?(?:, R\d+)*)\s*:\s*(\d+) Ø(\d+)", carte):
        for part in grp.split(", "):
            a_, _, b_ = part.partition("–")
            for i in range(int(a_[1:]), int((b_ or a_)[1:]) + 1):
                lues[f"R{i}"] = (int(n_), int(d_))
    for rep_, n_, d_ in RE.findall(r"(R\d+) (\d+) Ø(\d+)", carte):
        lues.setdefault(rep_, (int(n_), int(d_)))
    mal = [(r["rep"], r["n"], lues.get(r["rep"])) for r in fer7["renforts"] if r["n"] and lues.get(r["rep"]) != (r["n"], r["d"])]
    chk(f"A3 dalle 10,70 × 10,70 × 0,50 sans plot, Ø12 e 15 : pour chacun des {len(fer7['renforts'])} renforts, n = ⌈(A_s(m_dim) − 754) × "
        "largeur ÷ aire d'une barre⌉ avec A_s() du moteur Dalles (max(A_s,req ; A_s,min)), la largeur lue sur son rectangle ; la carte "
        "ARMATURES de l'écran écrit les barres de CHAQUE renfort sous SON repère",
        len(fer7["renforts"]) > 8 and not faux and not mal, (faux[:4], mal[:4]))
    chk("A3 un renfort de bande de la dalle 10,70 : sur toute la largeur de la bande (0,50 m), de la limite + l'ancrage, m_dim = 186,7 kNm/m "
        "(T19) ; n barres de Ø12",
        (lambda r: r["largeur"] == 0.5 and f"{r['m_dim']:.1f}" == "186.7" and r["n"] >= 2 and r["d"] == 12)(fer7["renforts"][0]),
        fer7["renforts"][0])
    # le besoin est max(A_s,req ; A_s,min) du moteur : une dalle de 1,00 m sous un treillis Ø8 e 25 en bas — A_s,min gouverne
    em = projet(h=1.0, armatures={"inf": {"x": {"d": 8, "e": 250}, "y": {"d": 8, "e": 250}, "xy": False}})
    ferm = AR.calculer(em, *efforts(em))
    gouv = [r for r in ferm["renforts"] if r["As_min"] > r["As_req"] + 1.0]
    chk("le besoin d'un renfort est max(A_s,req ; A_s,min) du moteur Dalles : h = 1,00 m, treillis du bas Ø8 e 25 — A_s,min gouverne des "
        "renforts du bas (le besoin vaut A_s,min, pas A_s,req)",
        gouv and all(abs(r["besoin"] - max(r["As_req"], r["As_min"])) < 1e-9 for r in ferm["renforts"])
        and all(abs(r["besoin"] - r["As_min"]) < 1e-9 for r in gouv), [(r["rep"], round(r["As_req"]), round(r["As_min"])) for r in ferm["renforts"][:4]])
    # l'étiquetage des zones rouges : 4-connexité (deux nœuds qui ne se touchent que par un coin font deux zones), numérotés
    # dans l'ordre du balayage ligne par ligne — scipy.ndimage.label
    lab, nb = AR.etiqueter(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], bool))
    chk("les zones rouges se regroupent en 4-connexité (gauche, droite, haut, bas ; un coin ne relie pas), numérotées dans l'ordre du "
        "balayage : quatre zones", nb == 4 and lab.tolist() == [[1, 0, 0], [0, 2, 0], [0, 0, 3], [4, 4, 0]], (nb, lab.tolist()))
    # A4 : la hauteur de béton
    e2 = projet(h=0.20)
    m2, ef2 = efforts(e2)
    fer2 = AR.calculer(e2, m2, ef2)
    textes = [t for t, _ in fer2["alertes"]]
    chk("A4 h = 0,20 m sur la dalle type : le moteur Dalles dit que d ne suffit pas pour m_dim (d_min > d) — pastille rouge « Hauteur de "
        "béton insuffisante : épaissir la dalle », la carte non conforme",
        fer2["hauteur"]["etat"] == "nok" and fer2["hauteur"]["hu_min"] > fer2["hauteur"]["d"] and fer2["etat"] == "nok"
        and ("Hauteur de béton insuffisante : épaissir la dalle", "r") in fer2["alertes"], (fer2["hauteur"], textes))
    tous = []
    for k in (dict(), dict(L=10.7), dict(plots=COULE), dict(L=10.7, plots=PREF), dict(h=0.20), dict(h=0.30),
              dict(armatures={"inf": {"x": {"d": 8, "e": 250}, "y": {"d": 8, "e": 250}, "xy": False}})):
        e_ = projet(**k)
        tous += [(t, n) for t, n in AR.calculer(e_, *efforts(e_))["alertes"]]
    epaissir = {t for t, _ in tous if "paissir" in t}
    chk("A4 aucune autre situation ne propose d'épaissir la dalle : sur sept dalles (h = 0,20 à 0,50, plots, treillis Ø8 e 25…), « épaissir » "
        "n'est dit que par la hauteur de béton et l'effort tranchant",
        epaissir <= {"Hauteur de béton insuffisante : épaissir la dalle", "Effort tranchant : τ > τ_adm — épaissir la dalle"}
        and fer["hauteur"]["etat"] == "ok" and not any("paissir" in t for t, _ in fer["alertes"]), sorted({t for t, _ in tous}))
    chk("la hauteur de béton de la dalle type : d_min = h_u,min (le plus exigeant des renforts et du treillis : un renfort en y) ≤ d = h − y_G "
        "= 42,0 cm, taux 61 %",
        fer["hauteur"]["etat"] == "ok" and abs(fer["hauteur"]["taux"] - fer["hauteur"]["hu_min"] / fer["hauteur"]["d"]) < 1e-12
        and round(fer["hauteur"]["taux"] * 100) == 61 and fer["hauteur"]["d"] == 42.0
        and next(r for r in fer["renforts"] if r["rep"] == fer["hauteur"]["rep"])["sens"] == "y", fer["hauteur"])
    # l'effort tranchant : V à d ÷ largeur au moteur Dalles (τ = v ÷ (0,75 × h)), V SCIA direct
    t1 = fer["tranchant"]["BX1"]
    bx1 = next(b for b in ef["bandes"] if b["nom"] == "BX1")
    fer_v = AR.calculer(el, m, ef, SC.rapports(dict(el, scia=dict(el["scia"], bx_v=120.0)), ef))
    chk("l'effort tranchant de BX1 : V à d = 94,0 kN (T14) ÷ 0,50 m → τ = 188,0 ÷ (0,75 × 500) = 0,50 N/mm² ≤ τ_adm,I du moteur ; V SCIA "
        "saisi (120 kN) va directement au moteur",
        abs(t1["V"] - bx1["V_d"]) < 1e-12 and f"{t1['V']:.1f}" == "94.0" and abs(t1["tau"] - t1["v"] / 375.0) < 1e-6 and t1["etat"] == "ok"
        and fer_v["tranchant"]["BX1"]["V"] == 120.0 and fer_v["tranchant"]["BX1"]["scia"] and abs(fer_v["tranchant"]["BX1"]["tau"] - 0.64) < 1e-6,
        (t1, fer_v["tranchant"]["BX1"]))
    # le Ø des renforts : automatique (le plus léger, à poids égal le plus gros), imposé
    par_d = {d: AR.calculer(projet(armatures={"renfort": d}), m, ef) for d in AR.DIAMETRES_RENFORT}
    choisi_ok = []
    for cle in AR.CLES:
        notes = {d: (any(r["serre"] for r in f["renforts"] if r["cle"] == cle), round(sum(r["poids"] for r in f["renforts"] if r["cle"] == cle), 3), -d)
                 for d, f in par_d.items()}
        choisi_ok.append(fer["diametres"][cle] == min(notes, key=notes.get))
    chk("le Ø automatique des renforts, face par face et direction par direction : le plus léger (Σ n × aire × longueur) de Ø12, Ø16, Ø20 — à "
        "poids égal le plus gros, un espacement trop serré en dernier ; imposé (Ø16) : tous les renforts en Ø16, ancrage 50 × 16 = 0,80 m",
        all(choisi_ok) and {r["d"] for r in par_d[16]["renforts"]} == {16} and {r["ancrage"] for r in par_d[16]["renforts"]} == {0.8},
        (fer["diametres"], choisi_ok))
    # châssis tourné : une bande prend le plus faible des deux directions de la face
    et = projet(L=12.0, grue={"beta": 30.0})
    mt, eft = efforts(et)
    fert = AR.calculer(et, mt, eft)
    chk("châssis tourné de 30° (dalle 12 × 12) : chaque bande prend le m_Rd le plus faible des deux directions de sa face (95,0 kNm/m en bas "
        "comme en haut) ; châssis droit : sa direction",
        fert["tourne"] and fert["mRd_bandes"]["inf_x"] == fert["mRd_bandes"]["inf_y"] == min(fert["res"]["mRd"]["inf_x"], fert["res"]["mRd"]["inf_y"])
        and not fer["tourne"] and fer["mRd_bandes"] == fer["res"]["mRd"]
        and all(r["cle_section"] in ("inf_y", "sup_y") for r in fert["renforts"] if r["type"] == "bande"), fert["mRd_bandes"])
    # le U de fermeture, les plots coulés, hors du moteur Dalles
    ec = projet(plots=COULE)
    ferc = AR.calculer(ec, *efforts(ec))
    hd = [AR.calculer(e_, *efforts(e_)) for e_ in (projet(dalle={"beton": "C45/55"}), projet(h=1.05))]
    chk("le U de fermeture (« Ø12 e 15 cm sur tout le pourtour (32,0 m) ») ; plots coulés en place : « Non vérifiés » nomme leur ferraillage, "
        "aucun renfort de bande dans un plot ; béton C45/55 ou h = 1,05 m : hors du moteur Dalles, les armatures « non vérifiées » (à vérifier, "
        "jamais conformes)",
        fer["u"] == "U de fermeture : Ø12 e 15 cm sur tout le pourtour (32,0 m)" and ferc["non_verifies"] == ["le poinçonnement", "le ferraillage des plots coulés en place"]
        and fer["non_verifies"] == ["le poinçonnement"] and ferc["plot_coule"]
        and all(h_["hors_domaine"] and h_["etat"] == "v" and not h_["renforts"] for h_ in hd)
        and "C45/55" in hd[0]["hors_domaine"] and "1,00 m" in hd[1]["hors_domaine"], (fer["u"], ferc["non_verifies"], [h_["hors_domaine"] for h_ in hd]))
    print("== 3. SCIA : E1 à E4 côté moteur")
    mo = SC.module(ef)
    sym = [SC.symetrique(projet(**k)) for k in (dict(), dict(grue={"ey": 0.5}), dict(grue={"beta": 90.0}), dict(grue={"beta": 45.0}),
                                                  dict(dalle={"Ly": 10.0}))]
    chk("E1 les valeurs grises sont celles du module (T14) : M+ le plus fort des bandes BX (71,6 kN·m) et BY, V à d (94,0 kN), m le plus "
        "négatif de la carte (−140,8 kN·m/m) ; « x = y » cochée d'office pour la dalle type (carrée, châssis carré centré), β = 90° ; pas "
        "pour une grue décalée, β = 45°, une dalle 8 × 10",
        (f"{mo['bx_m']:.1f}", f"{mo['bx_v']:.1f}", f"{mo['mx']:.1f}") == ("71.6", "94.0", "-140.8") and f2(mo["by_m"]) == f2(mo["bx_m"])
        and mo["bx_m"] == max(b["M_plus"] for b in ef["bandes"] if b["nom"].startswith("BX")) and mo["mx"] == ef["extremes"]["sup_x"]
        and sym == [True, False, True, False, False], (mo, sym))
    r0 = SC.rapports(el, ef, None)
    chk("E2 rien de saisi : r = 1 partout (les moments du module gardés), « à saisir », 3 champs à taper (x = y)",
        r0["echelle"] == {"inf_x": 1.0, "inf_y": 1.0, "sup_x": 1.0, "sup_y": 1.0} and not r0["saisi"] and not r0["complet"]
        and all(li["r"] is None for li in r0["lignes"]) and r0["a_taper"] == ["bx_m", "bx_v", "mx"], r0["echelle"])
    m7_ = SC.module(ef7)
    r11 = SC.rapports(dict(e7, scia=dict(e7["scia"], bx_m=m7_["bx_m"] * 1.10)), ef7)
    fer11 = AR.calculer(e7, m7, ef7, r11)
    pa11 = AR.principe(ef7, m7["contour"], fer11["res"]["mRd"], 0.60, {"inf_x": 1.1, "inf_y": 1.1}, fer11["mRd_bandes"])
    rb11 = [r for r in fer11["renforts"] if r["type"] == "bande"]
    rl11 = {c: [r for r in fer11["renforts"] if r["type"] == "local" and r["cle"] == c] for c in AR.CLES}
    chk("E3 dalle 10,70, bandes 0,50 : M+ SCIA = 1,10 × la valeur grise → r = 1,10, l'échelle du bas en x ET en y (x = y), le haut à 1 ; le "
        "principe du module est celui de T19 « SCIA = module × 1,10 en bas » (avec son mRd et son ancrage), et il a plus de barres",
        abs(r11["echelle"]["inf_x"] - 1.1) < 1e-12 and abs(r11["echelle"]["inf_y"] - 1.1) < 1e-9 and r11["echelle"]["sup_x"] == 1.0
        and _geo(rb11, rl11) == _geo(sorted(pa11["bandes"], key=lambda r: (r["cle"][:3] != "inf", r["cle"])), pa11["locaux"])
        and sum(r["n"] for r in fer11["renforts"]) > sum(r["n"] for r in fer7["renforts"]), r11["echelle"])
    def alerte(f):
        r_ = SC.rapports(dict(el, scia=dict(el["scia"], bx_m=mo["bx_m"] * f)), ef)
        return r_["lignes"][0]["alerte"], r_["alertes"]
    a130, a115, a084, a086 = alerte(1.30), alerte(1.15), alerte(0.84), alerte(0.86)
    chk("E4 r hors de 0,85 à 1,15 : l'alerte (« Écart module / SCIA : +30,0 % sur bandes BX : M+ le plus fort — vérifier le modèle SCIA (k, "
        "bandes, unités) ») ; r = 1,15 et 0,86 : pas d'alerte ; 0,84 : « −16,0 % »",
        a130[0] and a130[1] == ["Écart module / SCIA : +30,0 % sur bandes BX : M+ le plus fort — vérifier le modèle SCIA (k, bandes, unités)"]
        and not a115[0] and not a086[0] and a084[0] and "-16,0 %" in a084[1][0], (a130, a084))
    rs = SC.rapports(dict(el, scia=dict(el["scia"], sym=False, bx_m=80.0, by_m=60.0, mx=-150.0)), ef)
    rsig = SC.rapports(dict(el, scia=dict(el["scia"], sigma={"10": 100.0})), ef, [{"k": 10.0, "sigma_max": 110.3}])
    chk("« x = y » décochée : chaque direction a son r (BY : 60 ÷ 71,6) ; m SCIA négatif → r du haut ; σ max SCIA d'un k : comparé "
        "seulement (r = 100 ÷ 110,3), l'échelle des moments ne bouge pas",
        abs(rs["echelle"]["inf_y"] - 60.0 / mo["by_m"]) < 1e-12 and abs(rs["echelle"]["inf_x"] - 80.0 / mo["bx_m"]) < 1e-12
        and abs(rs["echelle"]["sup_x"] - 150.0 / -mo["mx"]) < 1e-12 and rs["echelle"]["sup_y"] == 1.0 and len(rs["a_taper"]) == 6
        and abs(next(s for s in rsig["sigma"] if s["k"] == 10.0)["r"] - 100 / 110.3) < 1e-12
        and rsig["echelle"] == {"inf_x": 1.0, "inf_y": 1.0, "sup_x": 1.0, "sup_y": 1.0}, (rs["echelle"], rsig["sigma"]))
    revue_15_11(el, m, ef, mo, fer)
    lot6_sections()


def couvert(bd, cov, face, ech=1.0):
    """Le plus fort dépassement (kN·m/m) du moment de la bande sur ce que reprennent ses armatures — écrit ICI, sans le dessin :
    le treillis, et sur chaque renfort le treillis + le renfort, atteint à un ancrage de chaque extrémité (au bord : tout de suite)."""
    s = np.asarray(bd["s"], float)
    res = np.full(len(s), cov["mRd"])
    for r in cov["renforts"]:
        an = r["ancrage"]
        k = np.minimum(np.ones(len(s)) if r["bord0"] else (s - r["s0"]) / an, np.ones(len(s)) if r["bord1"] else (r["s1"] - s) / an)
        k = np.where((s >= r["s0"]) & (s <= r["s1"]), np.clip(np.minimum(k, 1.0), 0.0, 1.0), 0.0)
        res = res + (r["mRd_pose"] - cov["mRd"]) * k
    m_ = (bd["Mmax"] if face == "inf" else -bd["Mmin"]) / bd["largeur"] * ech
    return float(np.max((m_ - res)[~bd["dans_plot"]]))


def lot6_sections():
    """Lot 6 (15/11/2026) : la SECTION DE CALCUL de chaque bande (saf/fondation_grue/sections.py) — le moteur Dalles calcule la
    bande comme une section b × h avec ce que le principe y pose ; elle redit le principe, et le contrôle."""
    from saf.fondation_grue import sections as SB
    print("== 3 ter. lot 6 : la section de calcul de chaque bande (moteur Dalles)")
    e7 = projet(L=10.7)
    m7, ef7 = efforts(e7)
    f7 = AR.calculer(e7, m7, ef7)
    s1 = f7["sections"][0]
    v1 = {v["cle"]: v for v in s1["verifications"]}
    g1 = s1["section"]["grandeurs"]
    r1 = s1["faces"]["inf"]["renfort"]
    a12 = AR.aire_barre(12)
    chk("L6-1 dalle 10,70 × 10,70 × 0,50, BX1 : la section 50 × 50 cm du moteur Dalles porte M⁺ = 93,4 kN·m (R1 : 186,7 × 0,50) et "
        "M⁻ = 17,3 kN·m, le treillis Ø12/150 + 4 Ø12 en bas, V à d = 84,1 kN ; ses vérifications : la hauteur (h_min = 36,0 ≤ 50 cm, 72 %), "
        "la face inférieure (A_s,néc = 724 ≤ A_s = 754 × 0,5 + 4 × 113,1 = 829 mm², 87 %), la face supérieure (377 ≤ 377 mm² : le minimum), "
        "τ (80 %) ; conforme, aucune alerte de plus",
        len(f7["sections"]) == 4 and (s1["b"], s1["h"]) == (50, 50) and list(v1) == ["hauteur", "inf_x", "sup_x", "tau"]
        and r1["rep"] == "R1" and (r1["n"], r1["d"]) == (4, 12) and f"{s1['faces']['inf']['M']:.1f}" == "93.4"
        and f"{g1['M_sup_x']:.2f}" == f"{s1['faces']['sup']['M']:.2f}" == "17.25" and f"{s1['V']:.1f}" == "84.1"
        and abs(v1["inf_x"]["limite"] - (754.0 * 0.5 + 4 * a12)) < 0.6 and f"{v1['inf_x']['valeur']:.1f}" == "723.8"
        and f"{v1['inf_x']['taux']:.3f}" == "0.873" and f"{v1['sup_x']['valeur']:.1f}" == "376.5" and f"{v1['hauteur']['valeur']:.1f}" == "36.0"
        and s1["section"]["faces"]["inf_x"]["libelle"] == "Ø12/150 + 4 Ø12" and s1["etat"] == "ok" and f7["etat"] == "ok"
        and not any(t.startswith("Bande") for t, _ in f7["alertes"]), (s1["faces"]["inf"]["M"], {k: (v["valeur"], v["limite"]) for k, v in v1.items()}))
    # la section redit le principe : même acier demandé, même acier posé, même τ (sur sept dalles)
    cas = {"type": projet(), "10,70": e7, "bandes 1,00": projet(largeur=1.0), "plots coulés": projet(plots=COULE),
           "plots préfabriqués": projet(plots=PREF), "tourné 30°": projet(L=10.7, grue={"beta": 30.0})}
    ecarts, pires, vus = [], {}, 0
    for nom, e_ in cas.items():
        m_, ef_ = efforts(e_)
        f_ = f7 if e_ is e7 else AR.calculer(e_, m_, ef_)
        pire = 0.0
        for s_, bd in zip(f_["sections"], ef_["bandes"]):
            for face in ("inf", "sup"):
                F = s_["faces"][face]
                v = next(x for x in s_["verifications"] if x["cle"] == F["cle"])
                if F["renfort"]:
                    vus += 1
                    if abs(v["valeur"] - F["renfort"]["besoin"] * s_["largeur"]) > 1e-6 or abs(v["limite"] - SB.pose_pm(F["renfort"]) * s_["largeur"]) > 1e-6:
                        ecarts.append((nom, s_["bande"], face))
                ech = (f_["echelle"].get(f"inf_{bd['nom'][1].lower()}", 1.0) if face == "inf" else f_["echelle"].get(f"sup_{s_['sens']}", 1.0))
                pire = max(pire, couvert(bd, s_["couverture"][face], face, ech))
            t_ = f_["tranchant"][s_["bande"]]
            vt = next(x for x in s_["verifications"] if x["cle"] == "tau")
            if abs(vt["taux"] - t_["taux"]) > 1e-12:
                ecarts.append((nom, s_["bande"], "tau"))
        pires[nom] = pire
    chk(f"L6-2 sur six dalles (type, 10,70, bandes 1,00, plots coulés et préfabriqués, châssis tourné de 30°), la section de chaque bande "
        f"redit le principe : l'acier demandé = A_s,néc du renfort × b, l'acier posé = (treillis + n barres) × b, le même τ ({vus} faces "
        "renforcées) ; et le moment de chaque bande reste partout sous ce que reprennent le treillis et ses renforts (le renfort atteint "
        "tout à un ancrage de son extrémité)",
        not ecarts and vus >= 20 and all(p_ <= 1e-9 for p_ in pires.values()), (ecarts[:4], pires))
    # SCIA : la section prend les moments × le rapport SCIA de sa face
    sc = SC.rapports(dict(e7, scia=dict(e7["scia"], mx=round(SC.module(ef7)["mx"] * 1.2, 4))), ef7)
    fs = AR.calculer(e7, m7, ef7, sc)
    chk("L6-3 SCIA saisi (m le plus négatif × 1,20) : la section prend M⁻ × 1,20 (17,25 → 20,7 kN·m), M⁺ ne bouge pas",
        abs(fs["echelle"]["sup_x"] - 1.2) < 1e-4 and abs(fs["sections"][0]["faces"]["sup"]["M"] - fs["echelle"]["sup_x"] * s1["faces"]["sup"]["M"]) < 1e-9
        and abs(fs["sections"][0]["faces"]["inf"]["M"] - s1["faces"]["inf"]["M"]) < 1e-9, fs["sections"][0]["faces"]["sup"]["M"])
    # le contrôle : la face opposée (réactions × 2,05) — le principe seul ne la voyait pas
    e_o = projet(L=10.7)
    e_o = PR.lire(dict(e_o, reactions=[{"nom": "", "P": [round(p_ * 2.05) for p_ in j]} for j in JEUX]))
    f_o = AR.calculer(e_o, *efforts(e_o))
    by = {s_["bande"]: s_ for s_ in f_o["sections"]}
    g_o = by["BY1"]["section"]["grandeurs"]
    chk("L6-4 réactions × 2,05 (dalle 10,70) : la hauteur tient (99 %) mais en haut des bandes BY le minimum 0,25·A_s,req de la face "
        "opposée (le bas) dépasse le treillis — le principe dimensionne chaque face seule et ne le voyait pas ; la section le voit : alerte "
        "rouge « Bande BY1, armature supérieure Y : sous le minimum 0,25·A_s,req de la face opposée (moteur Dalles) »",
        f_o["hauteur"]["etat"] == "ok" and by["BY1"]["causes"] == {"sup_y": "opposee"} and by["BY1"]["faces"]["sup"]["renfort"] is None
        and g_o["As_min_opposee_sup_y"] > g_o["As_sup_y"] > g_o["As_min_base"]
        and ("Bande BY1, armature supérieure Y : sous le minimum 0,25·A_s,req de la face opposée (moteur Dalles) — renforcer ou épaissir", "r")
        in f_o["alertes"] and f_o["etat"] == "nok", (by["BY1"]["causes"], [t for t, _ in f_o["alertes"]]))
    # h qui n'est pas un nombre entier de cm : d au cm inférieur, A_s,min au cm supérieur (comme le principe)
    e_h = projet(L=10.7, h=0.505)
    f_h = AR.calculer(e_h, *efforts(e_h))
    sh = f_h["sections"][0]
    vh = {v["cle"]: v for v in sh["verifications"]}
    rh = sh["faces"]["inf"]["renfort"]
    chk("L6-5 h = 0,505 m : la section est calculée à h = 50 cm (d côté sécurité) et dit A_s,min au cm supérieur (51 cm : 384 mm² sur la "
        "bande) ; la face renforcée redit le besoin du principe ; le treillis seul en haut est sous ce minimum — le principe le dit déjà "
        "(« Treillis sous le minimum »), la section ne le redit pas",
        sh["h"] == 50 and sh["plafond"]["h"] == 51 and f"{sh['plafond']['As_min']['sup_x']:.0f}" == "384"
        and abs(vh["inf_x"]["valeur"] - rh["besoin"] * 0.5) < 1e-6 and vh["sup_x"]["etat"] == "nok" and sh["causes"] == {"sup_x": "minimum"}
        and not any(t.startswith("Bande") for t, _ in f_h["alertes"]) and any(t.startswith("Treillis sous le minimum") for t, _ in f_h["alertes"]),
        (sh["plafond"], sh["causes"], [t for t, _ in f_h["alertes"]]))


def revue_15_11(el, m, ef, mo, fer):
    """Les défauts NON CONSERVATIFS trouvés par la relecture indépendante du 15/11/2026 (chacun a sa mutation)."""
    import copy
    print("== 3 bis. revue du 15/11/2026 : châssis à 90°, A_s,min, signe SCIA, « x = y », valeurs SCIA d'un autre modèle")
    # R1 : à β = 90°, la dalle type est la même qu'à β = 0 (châssis carré centré) : mêmes zones rouges, mêmes renforts ;
    # une bande BX y porte des barres en y
    e90 = projet(grue={"beta": 90.0})
    f90 = AR.calculer(e90, *efforts(e90))
    def compte(f):
        return ({c: int(f["rouge"][c].sum()) for c in AR.CLES},
                {c: sorted(r["type"] for r in f["renforts"] if r["cle"] == c) for c in AR.CLES})
    chk("revue R1 · châssis tourné de 90° (dalle type) : les mêmes nœuds rouges et les mêmes renforts qu'à 0° (16 nœuds rouges en bas en x, "
        "4 renforts locaux) ; les renforts de la bande BX1 vont en y — le sens d'une bande suit sa direction, pas son nom (§ 7.2, 7.3)",
        compte(f90) == compte(fer) and compte(fer)[0]["inf_x"] == 16
        and {r["sens"] for r in f90["renforts"] if r["bande"] == "BX1"} == {"y"}
        and {r["sens"] for r in fer["renforts"] if r["bande"] == "BX1"} == {"x"}, (compte(fer), compte(f90)))
    # R2 : h = 0,509 m — A_s,min au cm SUPÉRIEUR (51 cm : 768,1 mm²/m > Ø12 e 15 = 754,0)
    e51 = projet(h=0.509)
    f51 = AR.calculer(e51, *efforts(e51))
    chk("revue R2 · h = 0,509 m : A_s,min lu à 51 cm (768,14 mm²/m) — le treillis Ø12 e 15 (754,0) est SOUS le minimum, « Treillis sous "
        "le minimum » ; le besoin de chaque renfort compte ce minimum ; d reste lu à 50 cm",
        # (à la main : 0,26 × f_ctm ÷ f_yk × b × h = 0,26 × 0,30 × 30^(2/3) ÷ 500 × 1000 × 510 = 768,14 mm²/m)
        f2(f51["treillis"]["As_min"]["inf_x"]) == f2(0.26 * 0.30 * 30 ** (2 / 3) / 500 * 1000 * 510) == "768.14" and not any(f51["treillis"]["ok_min"].values()) and f51["etat"] == "nok"
        and any("Treillis sous le minimum" in a for a, _ in f51["alertes"]) and AR.h_moteur(e51) == 50 and AR.h_plafond(e51) == 51
        and all(r["besoin"] >= 768.14 - 1e-6 for r in f51["renforts"]) and all(fer["treillis"]["ok_min"].values()),
        (f51["treillis"]["As_min"], f51["treillis"]["ok_min"]))
    # R3 : un moment SCIA tapé avec le signe opposé (140,8 au lieu de −140,8) — r = |SCIA| ÷ |module|, alerte, renforts gardés
    def avec(**v):
        e_ = copy.deepcopy(el)
        e_["scia"].update(v)
        r_ = SC.rapports(e_, ef)
        return r_, AR.calculer(e_, m, ef, r_)
    r_bon, f_bon = avec(mx=mo["mx"])
    r_inv, f_inv = avec(mx=-mo["mx"])
    r_bas, f_bas = avec(bx_m=-mo["bx_m"])
    def n(f, face):
        return sum(1 for r in f["renforts"] if r["face"] == face)
    chk("revue R3 · m_x SCIA tapé +140,8 au lieu de −140,8 : r = 1,00 affiché et appliqué (|SCIA| ÷ |module|), les 12 renforts du haut gardés, l'alerte « Signe "
        "SCIA opposé à celui du module… » ; M+ SCIA négatif : les renforts du bas gardés ; aucune échelle ≤ 0",
        f2(r_inv["echelle"]["sup_x"]) == "1.00" and f2(next(li for li in r_inv["lignes"] if li["cle"] == "mx")["r"]) == "1.00"
        and n(f_inv, "sup") == n(f_bon, "sup") == 12 and n(f_bas, "inf") == n(f_bon, "inf")
        and r_inv["alertes"] and r_inv["alertes"][0].startswith("Signe SCIA opposé à celui du module sur dalle : m_x le plus négatif")
        and not r_bon["alertes"] and all(v > 0 for r_ in (r_inv, r_bas) for v in r_["echelle"].values()),
        (r_inv["echelle"], r_inv["alertes"], n(f_inv, "sup"), n(f_bas, "inf")))
    # R4 : « x = y » d'office seulement si appuis, plots et sol sont symétriques ; l'alerte d'une ligne y n'est plus cachée
    def rect(a, b, **k):
        e_ = projet(**k)
        e_["grue"]["rect"]["a"], e_["grue"]["rect"]["b"] = a, b
        return e_
    e_c2 = projet()
    e_c2["options"]["c2y"] = 0.2
    sy = [SC.symetrique(x) for x in (rect(0.5, 0.5), rect(0.70, 0.35), rect(0.60, 0.40), projet(plots=PREF), e_c2, projet(grue={"beta": 90.0}))]
    e7035 = rect(0.70, 0.35)
    m7035, ef7035 = efforts(e7035)
    mo7035 = SC.module(ef7035)
    e7035["scia"].update(sym=True, bx_m=mo7035["bx_m"], bx_v=mo7035["bx_v"], mx=mo7035["mx"])
    r7035 = SC.rapports(e7035, ef7035)
    chk("revue R4 · « x = y » cochée d'office : appuis 0,50 × 0,50, plots carrés, β = 90° — oui ; appuis 0,70 × 0,35 ou 0,60 × 0,40 à 0°, "
        "C2y ≠ C2x — non ; « x = y » cochée à la main sur les appuis 0,70 × 0,35 : la ligne y (r = 63,0 ÷ 77,1) garde son alerte",
        sy == [True, False, False, True, False, True]
        and any("bandes BY : M+ le plus fort" in a for a in r7035["alertes"]), (sy, r7035["alertes"]))
    # R5 : des valeurs SCIA tapées pour un AUTRE modèle (h changé ensuite) ne servent plus ; le nom ou la limite n'y font rien
    e_s = copy.deepcopy(el)
    e_s["scia"].update(bx_m=mo["bx_m"], bx_v=mo["bx_v"] * 0.5, mx=mo["mx"])
    e_s["scia"]["base"] = SC.signature(e_s)
    e_h = copy.deepcopy(e_s)
    e_h["dalle"]["h"] = 0.45
    e_n = copy.deepcopy(e_s)
    e_n["nom"], e_n["limite"] = "Autre", 2.0
    e_n["armatures"]["inf"]["x"]["d"] = 16
    r_s, r_h, r_n = SC.rapports(e_s, ef), SC.rapports(e_h, ef), SC.rapports(e_n, ef)
    chk("revue R5 · valeurs SCIA tapées puis h changé : elles ne servent plus (r = 1, V du module, « à ressaisir », étape non complète) ; "
        "changer le nom, la limite ou un treillis ne les périme pas",
        not r_s["perime"] and r_s["complet"] and r_s["V"]["x"] is not None
        and r_h["perime"] and not r_h["complet"] and r_h["V"] == {"x": None, "y": None}
        and r_h["echelle"] == {"inf_x": 1.0, "inf_y": 1.0, "sup_x": 1.0, "sup_y": 1.0} and r_h["alertes"][0] == SC.PERIME
        and not r_n["perime"] and r_n["complet"], (r_s["perime"], r_h["perime"], r_n["perime"], r_h["alertes"][:1]))
    # R6 : σ max SCIA hors de 0,85 à 1,15 — une alerte ; σ seul saisi compte comme saisi
    e_sig = copy.deepcopy(el)
    e_sig["scia"]["sigma"] = {"2": 60.0}
    r_sig = SC.rapports(e_sig, ef, [{"k": 2.0, "sigma_max": 46.0}])
    chk("revue R6 · σ max SCIA 60 kPa pour 46 du module (r = 1,30) : alerte « Écart module / SCIA : +30,4 % sur σ max, k = 2 MN/m³ » ; "
        "σ seul saisi : « saisi », pas « complet »",
        r_sig["saisi"] and not r_sig["complet"] and r_sig["sigma"][0]["alerte"]
        and any(a_.startswith("Écart module / SCIA : +30,4 % sur σ max, k = 2 MN/m³") for a_ in r_sig["alertes"]), r_sig["alertes"])
    # R7 : le U de fermeture — le treillis le plus FORT (A_s au mètre), pas le plus gros Ø
    e_u = copy.deepcopy(el)
    e_u["armatures"]["inf"]["x"] = {"d": 16, "e": 300}
    e_u["armatures"]["inf"]["xy"] = True
    f_u = AR.calculer(e_u, m, ef)
    chk("revue R7 · U de fermeture : treillis Ø16 e 30 (670 mm²/m) en bas en x et Ø12 e 15 (754) ailleurs — le U est en Ø12 e 15 (le plus "
        "fort), pas en Ø16 e 30 (le plus gros Ø)",
        f_u["u"].startswith("U de fermeture : Ø12 e 15 cm"), f_u["u"])
    # R8 : un plot coulé — « Ferraillage du plot : à prévoir » ; hors du moteur Dalles, « Non vérifiés » garde tout
    ec_ = projet(plots=COULE)
    f_c = AR.calculer(ec_, *efforts(ec_))
    ep_ = projet(plots=PREF)
    f_p = AR.calculer(ep_, *efforts(ep_))
    eh_ = projet(plots=COULE, dalle={"beton": "C45/55"})
    f_h = AR.calculer(eh_, *efforts(eh_))
    chk("revue R8 · plots coulés : la ligne « Ferraillage du plot : à prévoir (non dimensionné) » (§ 7.5) ; préfabriqués : rien ; C45/55 "
        "avec plots coulés : « Non vérifiés » nomme le poinçonnement, le ferraillage des plots ET les armatures",
        f_c["plot"] == "Ferraillage du plot : à prévoir (non dimensionné)" and f_p["plot"] is None
        and f_h["non_verifies"][:2] == ["le poinçonnement", "le ferraillage des plots coulés en place"]
        and f_h["non_verifies"][2].startswith("les armatures (") and f_h["plot"] == f_c["plot"], (f_c.get("plot"), f_h["non_verifies"]))
    # R10 : un renfort local qui entrerait dans l'encoche d'une dalle en L est coupé par le contour réel
    L_ = [list(p) for p in L_T7]
    ins = AR.dans_dalle({"x0": 6.0, "x1": 8.0, "y0": 8.0, "y1": 9.0, "aire": 0.04, "m_dim": 120.0}, "inf_x", L_T7)
    cheval = AR.dans_dalle({"x0": 6.0, "x1": 8.0, "y0": 6.5, "y1": 7.5, "aire": 0.04, "m_dim": 120.0}, "inf_x", L_T7)
    ins_y = AR.dans_dalle({"x0": 8.0, "x1": 9.0, "y0": 6.0, "y1": 8.0, "aire": 0.04, "m_dim": 120.0}, "inf_y", L_T7)
    eL = projet(dalle={"forme": "contour", "sommets": L_}, grue={"beta": 30.0})
    mL, efL = efforts(eL)
    orig = AR.principe
    def principe_essai(*a_, **k_):
        out = orig(*a_, **k_)
        out["locaux"]["inf_x"] = [{"x0": 6.0, "x1": 8.0, "y0": 8.0, "y1": 9.0, "aire": 0.04, "m_dim": 120.0}]
        return out
    AR.principe = principe_essai
    try:
        fL = AR.calculer(eL, mL, efL)
    finally:
        AR.principe = orig
    rl = [r for r in fL["renforts"] if r["type"] == "local" and r["cle"] == "inf_x"]
    chk("revue R10 · dalle en L (encoche x > 7, y > 7) : un renfort local en x de 6 à 8 m à y = 8–9 m s'arrête au bord à 7 m (longueur "
        "1,00 m) ; à cheval sur l'angle rentrant (y = 6,5–7,5) il garde ses 2 m ; en y, même règle — et le principe complet s'en sert",
        (ins["x0"], ins["x1"]) == (6.0, 7.0) and (cheval["x0"], cheval["x1"]) == (6.0, 8.0) and (ins_y["y0"], ins_y["y1"]) == (6.0, 7.0)
        and len(rl) == 1 and abs(rl[0]["x1"] - 7.0) < 1e-9 and abs(rl[0]["longueur"] - 1.0) < 1e-9, (ins, cheval, ins_y, rl[:1]))


def complets():
    print("== 4. la parité avec le calcul de référence (tests/fondation_grue_ref.py : principe_armatures, scipy.ndimage)")
    import fondation_grue_ref as REF
    cas = [("dalle en L 12 × 12, châssis tourné de 30°", projet(dalle={"forme": "contour", "sommets": [list(p) for p in L_T7]}, grue={"beta": 30.0}), (60.0, 80.0, 90.0, 70.0)),
           ("dalle 10 × 9, grue décalée (0,6 ; −0,4)", projet(L=10.0, dalle={"Ly": 9.0}, grue={"ex": 0.6, "ey": -0.4}), (140.0, 120.0, 100.0, 110.0)),
           ("dalle type, bandes 1,00, plots coulés", projet(largeur=1.0, plots=COULE), (75.0, 85.0, 65.0, 95.0)),
           ("dalle 10,70, plots préfabriqués", projet(L=10.7, plots=PREF), (97.3, 95.0, 97.3, 95.0))]
    for nom, el, mr in cas:
        m, ef = efforts(el)
        mRd = dict(zip(AR.CLES, mr))
        for anc, ech in ((0.80, None), (0.60, {"inf_x": 1.2, "sup_y": 0.9})):
            a = AR.principe(ef, m["contour"], mRd, anc, ech)
            b = REF.principe_armatures(ef, m["contour"], mRd, anc, ech)
            chk(f"{nom}, mRd {mr}, ancrage {anc} m{' , échelle SCIA' if ech else ''} : le principe du module = celui de la référence "
                f"({len(a['bandes'])} renforts de bande, {sum(len(v) for v in a['locaux'].values())} renforts locaux)",
                _geo(a["bandes"], a["locaux"]) == _geo(b["bandes"], b["locaux"]), (len(a["bandes"]), len(b["bandes"])))
    rng = np.random.default_rng(5)
    bon = True
    for _ in range(40):
        msk = rng.random((int(rng.integers(3, 15)), int(rng.integers(3, 15)))) < rng.uniform(0.2, 0.7)
        from scipy import ndimage
        la, na = AR.etiqueter(msk)
        lb, nb = ndimage.label(msk)
        bon &= na == nb and bool((la == lb).all())
    chk("l'étiquetage des zones rouges (numpy seul) est celui de scipy.ndimage.label : 40 grilles tirées au hasard, mêmes groupes, mêmes "
        "numéros", bon)
    print("== 5. A2 : changer un treillis ou la largeur des bandes ne refait pas le calcul du sol")
    from api_web import saf as M
    appels = []
    vrai = M.V.par_k

    def compte(*a, **k):
        appels.append(1)
        return vrai(*a, **k)
    M.V.par_k = compte
    try:
        e = M._etat_type()
        f = PR.lire(e["fondations"][0])
        c0 = M.calculer(f)
        z0 = M.armatures(f, c0)
        n0 = len(appels)
        f16 = PR.lire(dict(e["fondations"][0], armatures=dict(e["fondations"][0]["armatures"], inf={"x": {"d": 16, "e": 150}, "y": {"d": 16, "e": 150}, "xy": False})))
        t0 = time.perf_counter()
        z16 = M.armatures(f16, M.calculer(f16))
        dt = time.perf_counter() - t0
        f100 = PR.lire(dict(e["fondations"][0], bandes={"largeur": 1.0}))
        z100 = M.armatures(f100, M.calculer(f100))
    finally:
        M.V.par_k = vrai
    chk(f"A2 un treillis changé (bas Ø16 e 15) : les zones rouges et les renforts du bas suivent ({z0['zones']['inf_x']['aire']:.2f} → "
        f"{z16['zones']['inf_x']['aire']:.2f} m²), en {dt:.2f} s ; la largeur des bandes (1,00 m) : les renforts des bandes suivent ; le calcul du "
        "sol n'est PAS refait (aucun appel de verification.par_k après le premier)",
        n0 == 1 and len(appels) == 1 and z16["zones"]["inf_x"]["aire"] < z0["zones"]["inf_x"]["aire"] and dt < 1.0
        and {r["largeur"] for r in z100["renforts"] if r["type"] == "bande"} == {1.0}, (n0, len(appels)))


MUTATIONS = [
    ("armatures.py", '''            dep = (m > mRd_b[cle] + 1e-9) & ~bd["dans_plot"]''', '''            dep = (m > mRd_b[cle] + 1e-9)''',
     "un renfort de bande posé dans un plot coulé"),
    ("armatures.py", '''                s0, s1 = max(0.0, float(s[i0]) - ancrage), min(L, float(s[i1]) + ancrage)''',
     '''                s0, s1 = max(0.0, float(s[i0])), min(L, float(s[i1]))''', "un renfort de bande sans ancrage"),
    ("armatures.py", '''        rouge = (m > mRd[cle] + 1e-9) & g["actif"] & ~dans_bande[cle[-1]]''', '''        rouge = (m > mRd[cle] + 1e-9) & g["actif"]''',
     "une zone rouge dans une bande de sa direction"),
    ("armatures.py", '''        rouge = (m > mRd[cle] + 1e-9) & g["actif"] & ~dans_bande[cle[-1]]''',
     '''        rouge = (m > mRd[cle] + 1e-9) & g["actif"] & ~dans_bande["x"] & ~dans_bande["y"]''', "une bande qui couvre aussi les barres de l'autre direction"),
    ("armatures.py", '''                    for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):''',
     '''                    for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1), (j + 1, i + 1), (j - 1, i - 1), (j + 1, i - 1), (j - 1, i + 1)):''',
     "des zones rouges réunies en 8-connexité (scipy : 4)"),
    ("armatures.py", '''            r[lo] = max(LO, r[lo] - ancrage)''', '''            r[lo] = r[lo] - ancrage''', "un renfort local qui sort de la dalle"),
    ("armatures.py", '''    return int(math.ceil(renfort_pm * largeur / aire_barre(d_mm) - 1e-9))''', '''    return int(round(renfort_pm * largeur / aire_barre(d_mm)))''',
     "le nombre de barres arrondi au plus près"),
    ("armatures.py", '''    renfort = max(0.0, besoin - As_tr)''', '''    renfort = besoin''', "un renfort qui ne compte pas le treillis"),
    ("armatures.py", '''    besoin = max(g[f"besoin_{cle_sec}"], mini)''', '''    besoin = g[f"As_req_{cle_sec}"]''', "le besoin sans A_s,min"),
    ("armatures.py", '''    return {"mRd": {f: g[f"As_{f}"] / g[f"As_req_{f}"] for f in CLES}''', '''    return {"mRd": {f: 97.3 for f in CLES}''',
     "un mRd forfaitaire au lieu de celui du moteur Dalles"),
    ("armatures.py", '''"dist": round(POS.sur_couche(dessus_x, ty["d"]), 6)}]''', '''"dist": None}]''', "la couche y posée sur l'enrobage (d trop fort)"),
    ("armatures.py", '''        m = min(mRd[f"{face}_x"], mRd[f"{face}_y"])''', '''        m = max(mRd[f"{face}_x"], mRd[f"{face}_y"])''',
     "une bande de châssis tourné qui prend la direction la plus forte"),
    ("armatures.py", '''            note = (serre, round(poids, 3), -d)''', '''            note = (serre, -round(poids, 3), -d)''', "le Ø le plus lourd"),
    ("armatures.py", '''               "etat": "ok" if h_min <= h_cm + 1e-9 else "nok", "rep": rep_h}''', '''               "etat": "ok", "rep": rep_h}''',
     "une hauteur de béton insuffisante tue"),
    ("armatures.py", '''            V = abs(V_s)''', '''            pass''', "V SCIA ignoré"),
    ("armatures.py", '''    nv = ["le poinçonnement"] + (["le ferraillage des plots coulés en place"] if coule else [])''',
     '''    nv = ["le poinçonnement"]''', "le ferraillage des plots coulés dit vérifié"),
    ("armatures.py", '''    if h_plafond(el) > H_MAX_MOTEUR:''', '''    if False:''', "une dalle de plus de 1,00 m passée au moteur Dalles"),
    ("armatures.py", '''        v = V / bd["largeur"]''', '''        v = V''', "l'effort tranchant de la bande entière pris pour un mètre"),
    ("scia.py", '''    echelle = {"inf_x": ech(r["bx_m"]), "inf_y": ech(r["by_m"]), "sup_x": ech(r["mx"]), "sup_y": ech(r["my"])}''',
     '''    echelle = {"inf_x": ech(r["bx_m"]), "inf_y": ech(r["by_m"]), "sup_x": ech(r["bx_m"]), "sup_y": ech(r["by_m"])}''',
     "le haut recalé sur les bandes au lieu de la carte"),
    ("scia.py", '''                       "alerte": oppose or (e is not None and e > ECART + 1e-12)})''', '''                       "alerte": oppose or (e is not None and e > 0.5)})''',
     "l'écart de 15 % toléré jusqu'à 50 %"),
    ("scia.py", '''        for y, x in JUMELLES.items():
            s[y] = s[x]''', '''        for y, x in JUMELLES.items():
            pass''', "« x = y » qui ne recopie pas la direction y"),
    ("scia.py", '''    if abs(ch["a"] - ch["b"]) > 1e-9 or abs(g["ex"]) > 1e-9 or abs(g["ey"]) > 1e-9:''',
     '''    if abs(ch["a"] - ch["b"]) > 1e-9 or abs(g["ex"]) > 1e-9:''', "une grue décalée en y dite symétrique"),
    ("scia.py", '''        r = abs(s) / abs(m) if (s is not None and m not in (None, 0.0)) else None''', '''        r = abs(m) / abs(s) if (s is not None and m not in (None, 0.0)) else None''',
     "r = module ÷ SCIA"),
    # (revue du 15/11/2026 : les défauts non conservatifs)
    ("armatures.py", '''        nom_sens, sens = _sens_bande(bd["nom"]), axe_bande(bd)''', '''        nom_sens = sens = _sens_bande(bd["nom"])''',
     "le sens d'une bande tiré de son nom (β = 90° : barres en x sans contrôle)"),
    ("armatures.py", '''    if hc != h_moteur(el):''', '''    if False:''', "A_s,min lu au cm inférieur"),
    ("scia.py", '''        r = abs(s) / abs(m) if (s is not None''', '''        r = s / m if (s is not None''', "r gardant son signe (échelle négative)"),
    ("scia.py", '''                       "alerte": oppose or (e is not None''', '''                       "alerte": (e is not None''', "un signe opposé sans alerte"),
    ("scia.py", '''    return all(_memes_polygones(gr, [_echange(cx, cy, p) for p in gr]) for gr in groupes)''', '''    return True''',
     "des appuis allongés dits symétriques"),
    ("scia.py", '''    if abs(o["c1x"] - o["c1y"]) > 1e-12 or abs(o["c2x"] - o["c2y"]) > 1e-12:''', '''    if False:''', "des ressorts C2x ≠ C2y dits symétriques"),
    ("scia.py", '''    alertes = [texte(li) for li in lignes if li["alerte"] and montree(li)]''',
     '''    alertes = [texte(li) for li in lignes if li["alerte"] and not (s_ and li["cle"] in JUMELLES)]''', "l'alerte d'une ligne y toujours cachée"),
    ("scia.py", '''    vieux = perime(el)''', '''    vieux = False''', "des valeurs SCIA d'un autre modèle utilisées"),
    ("scia.py", '''HORS_MODELE = ("nom", "projet", "limite", "armatures",''', '''HORS_MODELE = ("nom", "projet", "armatures",''',
     "changer la limite périme les valeurs SCIA"),
    ("scia.py", '''    alertes += [f"Écart module / SCIA : {_pct((x['r'] - 1) * 100)} % sur σ max''', '''    alertes += [] and [f"Écart module / SCIA : {_pct((x['r'] - 1) * 100)} % sur σ max''',
     "un écart sur σ max sans alerte"),
    ("scia.py", '''            "saisi": any(sc[c] is not None for c in a_taper) or any(x["scia"] is not None for x in sigma),''',
     '''            "saisi": any(sc[c] is not None for c in a_taper),''', "σ seul saisi dit « non saisi »"),
    ("armatures.py", '''    fort = max(CLES, key=lambda c_: (round(res["As"][c_], 6), a[c_.split("_")[0]][c_[-1]]["d"]))''',
     '''    fort = max(CLES, key=lambda c_: (a[c_.split("_")[0]][c_[-1]]["d"], -a[c_.split("_")[0]][c_[-1]]["e"]))''', "le U du plus gros Ø (pas du plus fort)"),
    ("armatures.py", '''            "plot_coule": coule, "plot": PLOT_A_PREVOIR if coule else None, "diametres"''',
     '''            "plot_coule": coule, "plot": None, "diametres"''', "la ligne du plot coulé oubliée"),
    ("armatures.py", '''                "non_verifies": nv + ["les armatures (" + hd + ")"],''', '''                "non_verifies": ["les armatures (" + hd + ")"],''',
     "hors du moteur Dalles, « Non vérifiés » oublie le poinçonnement"),
    ("armatures.py", '''                r = dans_dalle(r, cle, contour)''', '''                pass''', "un renfort local qui entre dans l'encoche d'un L"),
    ("armatures.py", '''                    longueur, largeur = r["x1"] - r["x0"], r["y1"] - r["y0"]''', '''                    largeur, longueur = r["x1"] - r["x0"], r["y1"] - r["y0"]''',
     "la largeur d'un renfort local prise pour sa longueur"),
    # (lot 6 : la section de calcul de chaque bande)
    ("sections.py", '''    for f in faces.values():
        s[CLE_MOMENT[f["cle"]]] = max(0.0, float(f["M"]))''', '''    for f in [faces["inf"]]:
        s[CLE_MOMENT[f["cle"]]] = max(0.0, float(f["M"]))''', "la section sans M⁻ (la face opposée oubliée)"),
    ("sections.py", '''    for f in faces.values():
        s[CLE_MOMENT[f["cle"]]] = max(0.0, float(f["M"]))''', '''    for f in [faces["sup"]]:
        s[CLE_MOMENT[f["cle"]]] = max(0.0, float(f["M"]))''', "la section sans M⁺"),
    ("sections.py", '''        if r is None or not r["n"]:''', '''        if True:''', "la section sans le renfort de la bande"),
    ("sections.py", '''        return "opposee"''', '''        return "minimum"''', "la face opposée prise pour le minimum déjà dit"),
    ("sections.py", '''            if k == "minimum":''', '''            if True:''', "aucune alerte de la section"),
    ("sections.py", '''    if hc != h:''', '''    if False:''', "A_s,min de la section au cm inférieur"),
    ("sections.py", '''    return pose_pm(r) * fer["res"]["mRd"][cs] / fer["res"]["As"][cs]''', '''    return r["As_tr"] * fer["res"]["mRd"][cs] / fer["res"]["As"][cs]''',
     "la couverture sans le renfort"),
    ("sections.py", '''        e_ = ech.get(f"sup_{AR.axe_bande(bd)}", 1.0)''', '''        e_ = 1.0''', "M⁻ de la section sans le rapport SCIA"),
    ("armatures.py", '''    alertes += SB.alerte(out["sections"])''', '''    pass''', "le principe sans les alertes de la section"),
]
TEMOIN = ("armatures.py", "CLES = (", "CLES = (")


def muter(fichier, ancien, nouveau):
    """Les contrôles rapides rejoués sur une copie du moteur où ``ancien`` est remplacé : (texte trouvé, code de sortie)."""
    with tempfile.TemporaryDirectory() as tmp:
        for nom in os.listdir(RACINE):
            if nom in (".git", ".temoin_tests", "__pycache__", "node_modules", "saf"):
                continue
            os.symlink(os.path.join(RACINE, nom), os.path.join(tmp, nom))
        shutil.copytree(os.path.join(RACINE, "saf"), os.path.join(tmp, "saf"), ignore=shutil.ignore_patterns("__pycache__"))
        cible = os.path.join(tmp, "saf", "fondation_grue", fichier)
        src = open(cible, encoding="utf-8").read()
        if src.count(ancien) != 1:
            return False, None
        open(cible, "w", encoding="utf-8").write(src.replace(ancien, nouveau))
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--rapide", "--racine", tmp], capture_output=True, text=True, cwd=RACINE)
        return True, p.returncode


def main():
    t0 = time.time()
    rapides()
    print(f"   (contrôles rapides : {time.time() - t0:.1f} s)")
    if RAPIDE:
        print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
        return 1 if KO else 0
    complets()
    print("== 6. mutations")
    trouve, code = muter(*TEMOIN)
    chk("témoin : le moteur recopié sans rien changer passe les contrôles rapides", trouve and code == 0, code)
    for fichier, ancien, nouveau, nom in MUTATIONS:
        trouve, code = muter(fichier, ancien, nouveau)
        chk(f"mutation « {nom} » ({fichier}) : un contrôle rougit", trouve and code == 1,
            "texte absent ou répété" if not trouve else f"code {code} : AUCUN contrôle ne rougit")
    print("== 7. pureté")
    code = ("import sys\n"
            "class Refus:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('scipy', 'streamlit', 'pandas', 'matplotlib', 'reportlab', 'PIL', 'fitz', 'openpyxl'):\n"
            "            raise ImportError('refusé : ' + name)\n"
            "sys.meta_path.insert(0, Refus())\n"
            f"sys.path.insert(0, {RACINE!r})\n"
            "from saf.fondation_grue import projet as PR, verification as V, efforts as EF, armatures as AR, scia as SC\n"
            "el = PR.nouveau(); el['reactions'] = [{'nom': '', 'P': [350, 440, 440, 350]}, {'nom': '', 'P': [150, 150, 550, 550]}, "
            "{'nom': '', 'P': [135, 100, 800, 135]}]\n"
            "el = PR.lire(el); m = PR.modele(el)\n"
            "ef = EF.enveloppe(m, V.par_k(m, [2.0, 5.0, 10.0], (0.1, 0.1)), 0.44)\n"
            "fer = AR.calculer(el, m, ef, SC.rapports(el, ef))\n"
            "print(len(fer['renforts']), round(fer['res']['mRd']['inf_x'], 1))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    chk("le principe d'armatures se charge et calcule (dalle type : 28 renforts, mRd 97,3 kNm/m) avec numpy seul : scipy, streamlit, pandas, "
        "matplotlib, reportlab, openpyxl refusés", p.returncode == 0 and p.stdout.strip() == "28 97.3", (p.stdout.strip(), p.stderr[-300:]))
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom[:160], "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
